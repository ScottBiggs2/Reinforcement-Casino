#!/usr/bin/env bash
# Run a training batch script under an optional soft timeout; if the process stops due to
# SIGTERM/timeout before reaching target steps, optionally sbatch a continuation with
# DPO_RESUME=auto or GRPO_RESUME=auto (same run ids / mask paths as the initial job).
#
# Usage (from repo root, with the same exports as a normal training job):
#   export AUTO_RESUME_MODE=dense_dpo   # or sparse_dpo | dense_grpo | sparse_grpo
#   export AUTO_RESUME_SOFT_SECONDS=27000   # optional; omit to run inner without `timeout`
#   bash scripts/train_with_auto_resume.sh scripts/pipeline_stage_01_dense.sh
#
# Slurm: use as the job command (or --wrap) so continuations can sbatch the same line.
#
# Env:
#   AUTO_RESUME_MODE        (required) probe mode for scripts/training_resume_probe.py
#   AUTO_RESUME_SOFT_SECONDS  optional soft cap (seconds) before SIGTERM (leave headroom vs #SBATCH time)
#   MAX_AUTO_RESUME         max continuation jobs (default 8)
#   AUTO_RESUME_CONTINUE    incremented each continuation (default 0)
#   TRAIN_ENV               conda root for probe (default .../conda_envs/rl_casino)
#
# Optional integration: export USE_TRAIN_WITH_AUTO_RESUME=1 before sbatch of pipeline_stage_*.sh
# so the stage script exec's this wrapper (see top of those scripts).
#
set -euo pipefail

export USE_TRAIN_WITH_AUTO_RESUME=0

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"

INNER="${1:?Usage: train_with_auto_resume.sh <inner-script.sh> [args...]}"
shift

if [ -z "${AUTO_RESUME_MODE:-}" ]; then
  echo "ERROR: AUTO_RESUME_MODE must be set (dense_dpo|sparse_dpo|dense_grpo|sparse_grpo)" >&2
  exit 1
fi

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
MAX_AUTO_RESUME="${MAX_AUTO_RESUME:-8}"
CONTINUE="${AUTO_RESUME_CONTINUE:-0}"

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: $TRAIN_PY" >&2
  exit 1
fi

if (( CONTINUE > MAX_AUTO_RESUME )); then
  echo "ERROR: AUTO_RESUME_CONTINUE=${CONTINUE} exceeds MAX_AUTO_RESUME=${MAX_AUTO_RESUME}" >&2
  exit 1
fi

if [[ "${INNER}" = /* ]]; then
  INNER_ABS="${INNER}"
else
  INNER_ABS="${REPO_ROOT}/${INNER}"
fi

echo "train_with_auto_resume: mode=${AUTO_RESUME_MODE} continue=${CONTINUE}/${MAX_AUTO_RESUME} inner=${INNER_ABS}"

run_inner() {
  # Prevent pipeline_stage_* from re-exec'ing this wrapper when USE_TRAIN_WITH_AUTO_RESUME=1 was set for the outer job.
  USE_TRAIN_WITH_AUTO_RESUME=0 bash "${INNER_ABS}" "$@"
}

rc=0
if [ -n "${AUTO_RESUME_SOFT_SECONDS:-}" ]; then
  echo "Soft timeout: ${AUTO_RESUME_SOFT_SECONDS}s (SIGTERM at end; keep save_steps frequent + margin vs Slurm wall)"
  set +e
  timeout --signal=TERM --kill-after=120 "${AUTO_RESUME_SOFT_SECONDS}" \
    env USE_TRAIN_WITH_AUTO_RESUME=0 bash "${INNER_ABS}" "$@"
  rc=$?
  set -e
else
  set +e
  run_inner "$@"
  rc=$?
  set -e
fi

echo "Inner exit code: ${rc}"

export TRAINING_RESUME_PROBE_EXIT_POLICY=1
set +e
"${TRAIN_PY}" "${REPO_ROOT}/scripts/training_resume_probe.py" --mode "${AUTO_RESUME_MODE}" --json-only >/dev/null
pe=$?
set -e
echo "Probe exit (policy): ${pe}  (0=complete 10=resumable 11=no_ckpt 1=error)"

# Hard failures — never chain
if [[ "$rc" != 0 && "$rc" != 124 && "$rc" != 143 ]]; then
  exit "$rc"
fi

# Target reached
if [[ "$pe" == 0 ]]; then
  echo "train_with_auto_resume: target steps reached (or probe complete)."
  exit 0
fi

# Probe error
if [[ "$pe" == 1 ]]; then
  echo "ERROR: training_resume_probe failed (paths/env?)." >&2
  exit 1
fi

# No checkpoint to resume — do not spin (timeout before first save)
if [[ "$pe" == 11 ]]; then
  if [[ "$rc" == 124 || "$rc" == 143 ]]; then
    echo "train_with_auto_resume: stopped before a resumable checkpoint; not chaining." >&2
    exit "$rc"
  fi
  exit 0
fi

# pe==10 resumable
if [[ "$pe" != 10 ]]; then
  echo "train_with_auto_resume: unexpected probe exit ${pe}" >&2
  exit "$rc"
fi

if (( CONTINUE >= MAX_AUTO_RESUME )); then
  echo "train_with_auto_resume: resumable but MAX_AUTO_RESUME reached." >&2
  exit "$rc"
fi

NEXT=$((CONTINUE + 1))
echo "Submitting continuation ${NEXT}/${MAX_AUTO_RESUME}..."

RESUME_EXPORT="ALL,AUTO_RESUME_CONTINUE=${NEXT}"
case "${AUTO_RESUME_MODE}" in
  dense_dpo|sparse_dpo)
    RESUME_EXPORT+=",DPO_RESUME=auto"
    ;;
  dense_grpo|sparse_grpo)
    RESUME_EXPORT+=",GRPO_RESUME=auto"
    ;;
  *)
    echo "ERROR: bad AUTO_RESUME_MODE" >&2
    exit 1
    ;;
esac

WRAP_CMD="cd \"${REPO_ROOT}\" && bash \"${REPO_ROOT}/scripts/train_with_auto_resume.sh\" \"${INNER_ABS}\""

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "WARNING: not under Slurm; continuing inline." >&2
  export AUTO_RESUME_CONTINUE="${NEXT}"
  case "${AUTO_RESUME_MODE}" in
    dense_dpo|sparse_dpo) export DPO_RESUME=auto ;;
    dense_grpo|sparse_grpo) export GRPO_RESUME=auto ;;
  esac
  exec bash "${REPO_ROOT}/scripts/train_with_auto_resume.sh" "${INNER_ABS}" "$@"
fi

# shellcheck disable=SC2086
jid=$(sbatch --parsable \
  --export="${RESUME_EXPORT}" \
  --chdir="${REPO_ROOT}" \
  --wrap="${WRAP_CMD}")

echo "Continuation job: ${jid}"
exit 0

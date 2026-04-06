#!/usr/bin/env bash
# Full artifact reset for RL Casino pipeline runs (HPC scratch + optional repo logs).
# Order matches the operational plan: cancel jobs first (optional), then scratch, then logs.
# Paths match scripts/pipeline_common.sh defaults.
#
# Usage (from repo root on login node):
#   bash scripts/wipe_pipeline_artifacts.sh --run-id <PIPELINE_RUN_ID> --yes
#   bash scripts/wipe_pipeline_artifacts.sh --full-scratch --yes
#   bash scripts/wipe_pipeline_artifacts.sh --run-id <ID> --repo-logs --scancel-user --yes
#
# Options:
#   --run-id ID     Remove <ID> subdirs under train/mask/sparse/eval bases only.
#   --full-scratch  Remove entire rl_casino_train, rl_casino_masks, rl_casino_sparse_train, rl_casino_eval_runs under SCRATCH_USER_ROOT (destructive).
#   --repo-logs     Remove all files under <repo>/logs/ (not scratch).
#   --scancel-user  Run: scancel -u "$USER" (stop jobs before deleting paths they write to).
#   --include-hf-cache  Also rm -rf HF_DATASETS_CACHE_ROOT (slow to rebuild; default off).
#   --yes           Required to perform destructive operations (no prompt otherwise).
#
# Env: SCRATCH_USER_ROOT, TRAIN_OUT_BASE, MASK_OUT_BASE, SPARSE_OUT_BASE, EVAL_OUT_BASE, HF_DATASETS_CACHE_ROOT

set -euo pipefail

RUN_ID=""
FULL_SCRATCH=0
REPO_LOGS=0
SCANCEL_USER=0
INCLUDE_HF=0
CONFIRM=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="${2:?}"; shift 2 ;;
    --full-scratch) FULL_SCRATCH=1; shift ;;
    --repo-logs) REPO_LOGS=1; shift ;;
    --scancel-user) SCANCEL_USER=1; shift ;;
    --include-hf-cache) INCLUDE_HF=1; shift ;;
    --yes) CONFIRM=1; shift ;;
    -h|--help)
      grep '^#' "$0" | head -n 22 | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$CONFIRM" != "1" ]; then
  echo "Refusing: pass --yes to confirm destructive operations." >&2
  echo "See: bash scripts/wipe_pipeline_artifacts.sh --help" >&2
  exit 1
fi

if [ "$SCANCEL_USER" != "1" ] && [ "$FULL_SCRATCH" != "1" ] && [ -z "$RUN_ID" ] && [ "$REPO_LOGS" != "1" ] && [ "$INCLUDE_HF" != "1" ]; then
  echo "ERROR: specify at least one of --scancel-user, --run-id, --full-scratch, --repo-logs, --include-hf-cache" >&2
  exit 1
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_OUT_BASE="${TRAIN_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_train}"
MASK_OUT_BASE="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"
EVAL_OUT_BASE="${EVAL_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_eval_runs}"
HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE_ROOT:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

if [ "$SCANCEL_USER" = "1" ]; then
  echo "=== scancel -u ${USER} ==="
  scancel -u "${USER}" || true
fi

if [ "$FULL_SCRATCH" = "1" ]; then
  echo "=== Removing full scratch pipeline trees under ${SCRATCH_USER_ROOT} ==="
  rm -rf "${TRAIN_OUT_BASE}" "${MASK_OUT_BASE}" "${SPARSE_OUT_BASE}" "${EVAL_OUT_BASE}"
  echo "Removed: rl_casino_train, rl_casino_masks, rl_casino_sparse_train, rl_casino_eval_runs"
elif [ -n "$RUN_ID" ]; then
  echo "=== Removing RUN_ID=${RUN_ID} under scratch bases ==="
  rm -rf "${TRAIN_OUT_BASE}/${RUN_ID}" "${MASK_OUT_BASE}/${RUN_ID}" "${SPARSE_OUT_BASE}/${RUN_ID}" "${EVAL_OUT_BASE}/${RUN_ID}"
  echo "Removed subdirs under train/mask/sparse/eval bases."
fi

if [ "$INCLUDE_HF" = "1" ]; then
  echo "=== Removing HF datasets cache: ${HF_DATASETS_CACHE_ROOT} ==="
  rm -rf "${HF_DATASETS_CACHE_ROOT}"
fi

if [ "$REPO_LOGS" = "1" ]; then
  echo "=== Clearing repo logs: ${REPO_ROOT}/logs ==="
  mkdir -p "${REPO_ROOT}/logs"
  find "${REPO_ROOT}/logs" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  echo "Done."
fi

echo "=== wipe_pipeline_artifacts.sh finished ==="

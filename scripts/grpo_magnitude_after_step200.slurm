#!/usr/bin/env bash
# Magnitude follow-up for the GRPO 3-way orchestrator.
#
# Polls every 60s for the dense-GRPO step-200 delta file. Once present:
#   1. Builds a magnitude mask from |Δθ| accumulated up to GRPO_MAGNITUDE_TARGET_STEP via
#      src/warm_start/even_better_mask_finder.py (--method magnitude --target_step <N>).
#   2. Submits a sparse GRPO child via scripts/grpo_openr1_llama31_slurm.sh, using that mask.
#
# Submitted automatically by orchestrate_grpo_3way_focused.slurm. The required env vars come
# in via --export from the parent orchestrator (GRPO_DENSE_DELTA_DIR, MASK_GRPO_MAGNITUDE_STEP200,
# MODEL, GRPO_*).
#
# A few defaults are kept here so the script can be re-submitted standalone if the parent
# orchestrator dies before queueing this one.
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_mag_poll
#SBATCH --output=logs/grpo_mag_poll_%j.out
#SBATCH --error=logs/grpo_mag_poll_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

########################################
# Config (must agree with parent orchestrator if submitted via that path)
########################################

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
  exit 1
fi

# Inputs from parent orchestrator (with sensible standalone defaults).
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
export GRPO_DENSE_RUN_SLUG="${GRPO_DENSE_RUN_SLUG:-grpo_dense_openr1_steps800_with_deltas_v1}"
export GRPO_DENSE_DELTA_DIR="${GRPO_DENSE_DELTA_DIR:-${GRPO_DENSE_OUTPUT_BASE}/${GRPO_DENSE_RUN_SLUG}/deltas}"
export GRPO_MAGNITUDE_TARGET_STEP="${GRPO_MAGNITUDE_TARGET_STEP:-200}"
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# Sparse training knobs (mirror parent's sparse GraSP submission).
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-800}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-2048}"
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"
export GRPO_SPARSE_MAGNITUDE_RUN_NAME="${GRPO_SPARSE_MAGNITUDE_RUN_NAME:-grpo_sparse_magnitude_step${GRPO_MAGNITUDE_TARGET_STEP}_steps${GRPO_TARGET_STEPS}_v1}"

sanitize_model_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s//-/_}"
  s="${s,,}"
  s="$(echo "$s" | tr -c '[:alnum:]_' '_' )"
  while [[ "$s" == *"__"* ]]; do s="${s//__/_}"; done
  s="${s##_}"
  s="${s%%_}"
  echo "$s"
}

MODEL_SANITIZED="$(sanitize_model_name "$MODEL")"

# Default mask path if parent did not export one (must be deterministic so re-submits agree).
GRPO_DATASET_HF_DEFAULT="open-r1/OpenR1-Math-220k"
GRPO_DATASET_HF="${GRPO_DATASET_HF:-${GRPO_DATASET_HF_DEFAULT}}"
export MASK_GRPO_MAGNITUDE_STEP200="${MASK_GRPO_MAGNITUDE_STEP200:-${GRPO_MASK_DIR}/grpo_magnitude_${MODEL_SANITIZED}_${GRPO_DATASET_HF##*/}_sp${SPARSITY_PERCENT/./}_step${GRPO_MAGNITUDE_TARGET_STEP}.pt}"
mkdir -p "$(dirname "${MASK_GRPO_MAGNITUDE_STEP200}")"

########################################
# Polling loop
########################################

TARGET_DELTA_FILE="${GRPO_DENSE_DELTA_DIR}/deltas_step_${GRPO_MAGNITUDE_TARGET_STEP}.pt"
BASE_STATE_FILE="${GRPO_DENSE_DELTA_DIR}/base_state.pt"

# Generous wall-time-aware loop: poll every 60s for up to ~7h45m, well under the 8h #SBATCH cap.
POLL_INTERVAL_SECS="${GRPO_MAG_POLL_INTERVAL_SECS:-60}"
POLL_MAX_SECS="${GRPO_MAG_POLL_MAX_SECS:-$(( 7 * 3600 + 45 * 60 ))}"
START_TS=$(date +%s)

echo "Polling for ${TARGET_DELTA_FILE}"
echo "  interval=${POLL_INTERVAL_SECS}s max=${POLL_MAX_SECS}s"
echo "  base_state expected at: ${BASE_STATE_FILE}"

while true; do
  if [ -f "${TARGET_DELTA_FILE}" ] && [ -f "${BASE_STATE_FILE}" ]; then
    echo "  ✓ Found ${TARGET_DELTA_FILE} (and base_state.pt) at $(date -Is)."
    break
  fi
  now=$(date +%s)
  elapsed=$(( now - START_TS ))
  if (( elapsed > POLL_MAX_SECS )); then
    echo "ERROR: timed out after ${elapsed}s waiting for ${TARGET_DELTA_FILE}" >&2
    echo "  Either dense GRPO has not reached step ${GRPO_MAGNITUDE_TARGET_STEP} yet, or delta logging is misconfigured." >&2
    exit 1
  fi
  if (( elapsed % 600 < POLL_INTERVAL_SECS )); then
    echo "  ... still waiting (elapsed=${elapsed}s)"
    if [ -d "${GRPO_DENSE_DELTA_DIR}" ]; then
      ls "${GRPO_DENSE_DELTA_DIR}" 2>/dev/null | head -n 8 | sed 's/^/      /'
    fi
  fi
  sleep "${POLL_INTERVAL_SECS}"
done

########################################
# Build magnitude mask
########################################

echo ""
echo "=== Building magnitude mask → ${MASK_GRPO_MAGNITUDE_STEP200}"
"$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "${GRPO_DENSE_DELTA_DIR}" \
  --method magnitude \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --target_step "${GRPO_MAGNITUDE_TARGET_STEP}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --output_file "${MASK_GRPO_MAGNITUDE_STEP200}" \
  --force_cpu

if [ ! -f "${MASK_GRPO_MAGNITUDE_STEP200}" ]; then
  echo "ERROR: magnitude mask was not written; aborting before sparse submit." >&2
  exit 1
fi

########################################
# Submit sparse GRPO child with the magnitude mask
########################################

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

echo ""
echo "=== Submitting sparse GRPO with magnitude mask ==="
SPARSE_MAG_EXPORT="ALL"
SPARSE_MAG_EXPORT+=",GRPO_MODE=sparse"
SPARSE_MAG_EXPORT+=",GRPO_MASK=${MASK_GRPO_MAGNITUDE_STEP200}"
SPARSE_MAG_EXPORT+=",GRPO_RUN_NAME=${GRPO_SPARSE_MAGNITUDE_RUN_NAME}"
SPARSE_MAG_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
SPARSE_MAG_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
SPARSE_MAG_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
SPARSE_MAG_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
SPARSE_MAG_EXPORT+=",WANDB_MODE=online"
SPARSE_MAG_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
SPARSE_MAG_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
SPARSE_MAG_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
SPARSE_MAG_EXPORT+=",MODEL=${MODEL}"
SPARSE_MAG_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"

mag_jname="$(orch_sanitize_job_name "grpo_sp_mag_${GRPO_SPARSE_MAGNITUDE_RUN_NAME}")"
mag_jid=$(sbatch --parsable \
  --job-name="${mag_jname}" \
  --export="${SPARSE_MAG_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
echo "GRPO sparse magnitude (step ${GRPO_MAGNITUDE_TARGET_STEP}): ${mag_jid}"

echo ""
echo "=== Done ==="
echo "Magnitude mask: ${MASK_GRPO_MAGNITUDE_STEP200}"
echo "Monitor: squeue -u ${USER:-<user>}"

#!/usr/bin/env bash
# One-off orchestrator: GRPO Open-R1 3-way focused sweep.
#   1. Regenerate GraSP element/SNR mask (1D element-wise + per_weight SNR + log1p).
#   2. Submit dense GRPO (with delta logging at step 50/100/150/200; continues to GRPO_TARGET_STEPS).
#      This dense run doubles as the comparison condition AND the magnitude source.
#   3. Submit sparse GRPO with the GraSP-elem-SNR mask (immediately, mask is on disk by this point).
#   4. Submit magnitude follow-up poller (scripts/grpo_magnitude_after_step200.slurm) — that job
#      waits for deltas_step_200.pt under the dense run dir, then builds the magnitude mask and
#      submits the sparse GRPO with it.
#
# Submit from repo root:
#   sbatch scripts/orchestrate_grpo_3way_focused.slurm
#
# Override knobs by exporting env vars before sbatch (see "Config" section).
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=orch_grpo_3way
#SBATCH --output=logs/orch_grpo_3way_%j.out
#SBATCH --error=logs/orch_grpo_3way_%j.err

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
# Config (override before sbatch)
########################################

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

# Output bases (must agree with grpo_openr1_llama31_slurm.sh defaults).
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
export GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"
mkdir -p "${GRPO_MASK_DIR}"

# Model + dataset (Open-R1 Math defaults match grpo_openr1_llama31_slurm.sh).
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

# Sparsity + global pool floor (matches the rest of the pipeline).
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# Sparse GRPO training children (dense GRPO stays on default #SBATCH / H200 in grpo_openr1_llama31_slurm.sh).
export ORCH_TRAIN_PARTITION_SPARSE="${ORCH_TRAIN_PARTITION_SPARSE:-gpu}"
export ORCH_TRAIN_GRES_SPARSE="${ORCH_TRAIN_GRES_SPARSE:-gpu:a100:1}"
export ORCH_TRAIN_TIME_GRPO="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"

# Mask calibration knobs.
export GRPO_GRASP_OBJECTIVE="${GRPO_GRASP_OBJECTIVE:-lm}"
export GRPO_GRASP_PREFERENCE_BETA="${GRPO_GRASP_PREFERENCE_BETA:-1.0}"
export GRPO_N_SAMPLES="${GRPO_N_SAMPLES:-256}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-1}"

# SNR knobs (per_weight + log1p — the focused setting requested).
export ORCH_SCORE_SNR_MODE="${ORCH_SCORE_SNR_MODE:-off}"
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-96}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

# Training knobs (forwarded to sbatched grpo_openr1_llama31_slurm.sh children).
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-1000}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
# Align with docs/hyperparams/open_r1_llama31.yaml + grpo_openr1_llama31_slurm.sh (1024, not 2048).
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-1024}"
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"

# Delta logging schedule (only applied to the dense GRPO child; magnitude is built from step 200).
export GRPO_DELTA_LOG_INTERVAL="${GRPO_DELTA_LOG_INTERVAL:-50}"
export GRPO_DELTA_LOG_END_STEP="${GRPO_DELTA_LOG_END_STEP:-200}"
export GRPO_MAGNITUDE_TARGET_STEP="${GRPO_MAGNITUDE_TARGET_STEP:-200}"

# Run slugs / names. The dense run_slug defines where deltas land:
#   <GRPO_DENSE_OUTPUT_BASE>/<GRPO_DENSE_RUN_SLUG>/deltas/{base_state,deltas_step_<N>}.pt
export GRPO_DENSE_RUN_SLUG="${GRPO_DENSE_RUN_SLUG:-grpo_dense_openr1_steps${GRPO_TARGET_STEPS}_with_deltas_v1}"
export GRPO_SPARSE_GRASP_RUN_NAME="${GRPO_SPARSE_GRASP_RUN_NAME:-grpo_sparse_grasp_${GRPO_GRASP_OBJECTIVE}_elem_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}_steps${GRPO_TARGET_STEPS}_v1}"
export GRPO_SPARSE_MAGNITUDE_RUN_NAME="${GRPO_SPARSE_MAGNITUDE_RUN_NAME:-grpo_sparse_magnitude_step${GRPO_MAGNITUDE_TARGET_STEP}_steps${GRPO_TARGET_STEPS}_v1}"

# Magnitude poller wall (must outlast time-to-step-200 of dense GRPO; default 8h).
export GRPO_MAG_POLLER_WALL="${GRPO_MAG_POLLER_WALL:-08:00:00}"

########################################
# Preflight
########################################

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
  exit 1
fi

"$TRAIN_PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available; this orchestrator must run on a GPU node.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: device_count={torch.cuda.device_count()} device0={torch.cuda.get_device_name(0)!r}")
PY

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

########################################
# Derived paths (mask + dense run dir)
########################################

# GraSP element/SNR mask path. Filename mirrors the convention used by
# orchestrate_masks_then_queue_dpo_grpo.slurm so downstream tooling can find it.
export MASK_GRPO_GRASP_ELEM_SNR="${GRPO_MASK_DIR}/grpo_grasp_${GRPO_GRASP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_${MODEL_SANITIZED}_${GRPO_DATASET_HF##*/}_elem_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}.pt"

# Dense run dir (must match GRPO_train.py: run_dir = output_base_dir / run_slug).
export GRPO_DENSE_RUN_DIR="${GRPO_DENSE_OUTPUT_BASE}/${GRPO_DENSE_RUN_SLUG}"
export GRPO_DENSE_DELTA_DIR="${GRPO_DENSE_RUN_DIR}/deltas"

# Magnitude mask path (poller writes here; consumer reads from here).
export MASK_GRPO_MAGNITUDE_STEP200="${GRPO_MASK_DIR}/grpo_magnitude_${MODEL_SANITIZED}_${GRPO_DATASET_HF##*/}_sp${SPARSITY_PERCENT/./}_step${GRPO_MAGNITUDE_TARGET_STEP}.pt"

echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL=${MODEL} (sanitized=${MODEL_SANITIZED})"
echo "GRPO_MASK_DIR=${GRPO_MASK_DIR}"
echo "GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE}"
echo "GRPO_DENSE_RUN_SLUG=${GRPO_DENSE_RUN_SLUG}"
echo "GRPO_DENSE_DELTA_DIR=${GRPO_DENSE_DELTA_DIR}"
echo "MASK_GRPO_GRASP_ELEM_SNR=${MASK_GRPO_GRASP_ELEM_SNR}"
echo "MASK_GRPO_MAGNITUDE_STEP200=${MASK_GRPO_MAGNITUDE_STEP200}"
echo "Sparsity=${SPARSITY_PERCENT}%  N_SAMPLES=${GRPO_N_SAMPLES}  STEPS=${GRPO_TARGET_STEPS}"
echo "Delta logging: every ${GRPO_DELTA_LOG_INTERVAL} up to ${GRPO_DELTA_LOG_END_STEP}"

########################################
# 1) Generate GraSP element/SNR mask
########################################

SNR_ARGS=()
if [ "${ORCH_SCORE_SNR_MODE}" != "off" ]; then
  SNR_ARGS=( --score-snr "${ORCH_SCORE_SNR_MODE}"
             --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}"
             --score-snr-eps "${ORCH_SCORE_SNR_EPS}"
             --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}" )
  if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
    SNR_ARGS+=( --score-snr-allow-large-ram )
  fi
fi

echo ""
echo "=== Generating GRPO GRaSP element/SNR mask → ${MASK_GRPO_GRASP_ELEM_SNR}"
set +e
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method grasp \
  --grasp-objective "${GRPO_GRASP_OBJECTIVE}" \
  --dataset_name "${GRPO_DATASET_HF}" \
  --n_samples "${GRPO_N_SAMPLES}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --seed 42 \
  --grasp-preference-beta "${GRPO_GRASP_PREFERENCE_BETA}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  "${SNR_ARGS[@]}" \
  --mask-granularity element \
  --output "${MASK_GRPO_GRASP_ELEM_SNR}"
mask_rc=$?
set -e

if [ "${mask_rc}" -ne 0 ] || [ ! -f "${MASK_GRPO_GRASP_ELEM_SNR}" ]; then
  echo "ERROR: GraSP mask generation failed (exit ${mask_rc}); not submitting sparse GraSP child." >&2
fi

########################################
# 2) Submit children (dense + sparse GraSP + magnitude poller)
########################################

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

echo ""
echo "=== Submitting GRPO dense child (delta logging through step ${GRPO_DELTA_LOG_END_STEP}) ==="
DENSE_EXPORT="ALL"
DENSE_EXPORT+=",GRPO_MODE=dense"
DENSE_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
DENSE_EXPORT+=",GRPO_RUN_SLUG=${GRPO_DENSE_RUN_SLUG}"
DENSE_EXPORT+=",GRPO_DELTA_LOG_INTERVAL=${GRPO_DELTA_LOG_INTERVAL}"
DENSE_EXPORT+=",GRPO_DELTA_LOG_END_STEP=${GRPO_DELTA_LOG_END_STEP}"
DENSE_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
DENSE_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
DENSE_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
DENSE_EXPORT+=",WANDB_MODE=online"
DENSE_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
DENSE_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
DENSE_EXPORT+=",GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE}"
DENSE_EXPORT+=",MODEL=${MODEL}"
DENSE_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"

dense_jname="$(orch_sanitize_job_name "grpo_dense_${GRPO_DENSE_RUN_SLUG}")"
dense_jid=$(sbatch --parsable \
  --job-name="${dense_jname}" \
  --export="${DENSE_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
echo "GRPO dense (delta-logged): ${dense_jid}"

if [ -f "${MASK_GRPO_GRASP_ELEM_SNR}" ]; then
  echo ""
  echo "=== Submitting sparse GRPO with GraSP-elem-SNR mask ==="
  SPARSE_GRASP_EXPORT="ALL"
  SPARSE_GRASP_EXPORT+=",GRPO_MODE=sparse"
  SPARSE_GRASP_EXPORT+=",GRPO_MASK=${MASK_GRPO_GRASP_ELEM_SNR}"
  SPARSE_GRASP_EXPORT+=",GRPO_RUN_NAME=${GRPO_SPARSE_GRASP_RUN_NAME}"
  SPARSE_GRASP_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
  SPARSE_GRASP_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
  SPARSE_GRASP_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
  SPARSE_GRASP_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
  SPARSE_GRASP_EXPORT+=",WANDB_MODE=online"
  SPARSE_GRASP_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
  SPARSE_GRASP_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
  SPARSE_GRASP_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
  SPARSE_GRASP_EXPORT+=",MODEL=${MODEL}"
  SPARSE_GRASP_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"

  grasp_jname="$(orch_sanitize_job_name "grpo_sp_grasp_${GRPO_SPARSE_GRASP_RUN_NAME}")"
  grasp_jid=$(sbatch --parsable \
    --partition="${ORCH_TRAIN_PARTITION_SPARSE}" \
    --nodes=1 \
    --ntasks=1 \
    --gres="${ORCH_TRAIN_GRES_SPARSE}" \
    --mem=128G \
    --time="${ORCH_TRAIN_TIME_GRPO}" \
    --cpus-per-task=8 \
    --job-name="${grasp_jname}" \
    --export="${SPARSE_GRASP_EXPORT}" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
  echo "GRPO sparse GraSP-elem-SNR: ${grasp_jid}"
else
  echo "WARNING: skipping sparse GraSP submit (mask file not on disk)."
fi

echo ""
echo "=== Submitting magnitude follow-up poller (waits for ${GRPO_DENSE_DELTA_DIR}/deltas_step_${GRPO_MAGNITUDE_TARGET_STEP}.pt) ==="
POLL_EXPORT="ALL"
POLL_EXPORT+=",MODEL=${MODEL}"
POLL_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"
POLL_EXPORT+=",GRPO_DENSE_DELTA_DIR=${GRPO_DENSE_DELTA_DIR}"
POLL_EXPORT+=",GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE}"
POLL_EXPORT+=",GRPO_DENSE_RUN_SLUG=${GRPO_DENSE_RUN_SLUG}"
POLL_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
POLL_EXPORT+=",GRPO_MASK_DIR=${GRPO_MASK_DIR}"
POLL_EXPORT+=",MASK_GRPO_MAGNITUDE_STEP200=${MASK_GRPO_MAGNITUDE_STEP200}"
POLL_EXPORT+=",GRPO_MAGNITUDE_TARGET_STEP=${GRPO_MAGNITUDE_TARGET_STEP}"
POLL_EXPORT+=",GRPO_SPARSE_MAGNITUDE_RUN_NAME=${GRPO_SPARSE_MAGNITUDE_RUN_NAME}"
POLL_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
POLL_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
POLL_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
POLL_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
POLL_EXPORT+=",WANDB_MODE=online"
POLL_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
POLL_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
POLL_EXPORT+=",SPARSITY_PERCENT=${SPARSITY_PERCENT}"
POLL_EXPORT+=",MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}"

poll_jid=$(sbatch --parsable \
  --time="${GRPO_MAG_POLLER_WALL}" \
  --export="${POLL_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_magnitude_after_step200.slurm")
echo "Magnitude follow-up poller: ${poll_jid}"

echo ""
echo "=== Done — all 3-way components queued. ==="
echo "Dense run dir (deltas land here):  ${GRPO_DENSE_RUN_DIR}"
echo "GraSP mask: ${MASK_GRPO_GRASP_ELEM_SNR}"
echo "Magnitude mask (will be written by poller): ${MASK_GRPO_MAGNITUDE_STEP200}"
echo "Monitor: squeue -u ${USER:-<user>}"

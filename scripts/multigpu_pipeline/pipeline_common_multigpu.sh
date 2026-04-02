# shellcheck shell=bash
# Shared config for multi-GPU pipeline entrypoints.
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "$REPO_ROOT"

# RUN_ID handling mirrors scripts/pipeline_common.sh
if [ -n "${PIPELINE_RUN_ID:-}" ]; then
  export RUN_ID="${PIPELINE_RUN_ID}"
elif [ -n "${RUN_ID_OVERRIDE:-}" ]; then
  export RUN_ID="${RUN_ID_OVERRIDE}"
else
  export RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
fi
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-$RUN_ID}"

# Envs (reuse the same envs as the baseline pipeline)
TRAIN_ENV="${TRAIN_ENV:-/scratch/biggs.s/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"

TRAIN_OUT_BASE="${TRAIN_OUT_BASE:-/scratch/biggs.s/rl_casino_train}"
MASK_OUT_BASE="${MASK_OUT_BASE:-/scratch/biggs.s/rl_casino_masks}"
SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-/scratch/biggs.s/rl_casino_sparse_train}"
EVAL_OUT_BASE="${EVAL_OUT_BASE:-/scratch/biggs.s/rl_casino_eval_runs}"

mkdir -p "$TRAIN_OUT_BASE" "$MASK_OUT_BASE" "$SPARSE_OUT_BASE" "$EVAL_OUT_BASE" logs

# Model/dataset kept identical to the default pipeline unless overridden.
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# Keep the interface simple: one dataset key for the dense DPO stage.
# (The single-GPU pipeline supports an array; we can extend later if needed.)
DPO_DATASET_KEY="${DPO_DATASET_KEY:-tulu3}"

# Multi-GPU settings
# MULTIGPU_NGPUS can be an integer, or "auto" to use the number of visible GPUs.
MULTIGPU_NGPUS="${MULTIGPU_NGPUS:-auto}"
MULTIGPU_GPU_TYPE="${MULTIGPU_GPU_TYPE:-h200}"
MULTIGPU_RESERVATION="${MULTIGPU_RESERVATION:-}"  # optional

# Paper-parity defaults for DPO on Tulu3 (arXiv:2505.11711 Appendix B)
# Effective batch = per_device_bs * grad_accum * world_size
SEQ_LEN="${SEQ_LEN:-2048}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR_PEAK="${LR_PEAK:-5e-7}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
# For dense DPO duration you can choose either:
# - NUM_STEPS_DPO (explicit steps), or
# - NUM_EPOCHS (epochs; if both are set, epochs wins).
NUM_STEPS_DPO="${NUM_STEPS_DPO:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"

# Warm-start artifact schedule (keep as in baseline pipeline unless overridden)
DELTA_LOG_INTERVAL="${DELTA_LOG_INTERVAL:-50}"
DELTA_LOG_END_STEP="${DELTA_LOG_END_STEP:-200}"

# Timeouts
TRAIN_TIMEOUT_PER_DATASET="${TRAIN_TIMEOUT_PER_DATASET:-$((7 * 60 * 60 + 30 * 60))}"

pipeline_setup_multigpu() {
  echo "Multi-GPU pipeline RUN_ID=${RUN_ID}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Training env: ${TRAIN_ENV}"
  if [ ! -x "$TRAIN_PY" ]; then
    echo "ERROR: TRAIN_PY not found at ${TRAIN_PY}" >&2
    exit 1
  fi
  export PATH="${TRAIN_ENV}/bin:${PATH}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1

  echo "Multi-GPU settings:"
  echo "  MULTIGPU_NGPUS=${MULTIGPU_NGPUS}"
  echo "  MULTIGPU_GPU_TYPE=${MULTIGPU_GPU_TYPE}"
  echo "  MULTIGPU_RESERVATION=${MULTIGPU_RESERVATION:-<none>}"

  echo "DPO parity knobs:"
  echo "  SEQ_LEN=${SEQ_LEN}"
  echo "  PER_DEVICE_BS=${PER_DEVICE_BS}"
  echo "  GRAD_ACCUM=${GRAD_ACCUM}"
  echo "  LR_PEAK=${LR_PEAK}"
  echo "  WARMUP_RATIO=${WARMUP_RATIO}"
  echo "  WEIGHT_DECAY=${WEIGHT_DECAY}"
  echo "  NUM_STEPS_DPO=${NUM_STEPS_DPO:-<unset>}"
  echo "  NUM_EPOCHS=${NUM_EPOCHS:-<unset>}"

  echo "Warm-start delta schedule:"
  echo "  DELTA_LOG_INTERVAL=${DELTA_LOG_INTERVAL}"
  echo "  DELTA_LOG_END_STEP=${DELTA_LOG_END_STEP}"
}


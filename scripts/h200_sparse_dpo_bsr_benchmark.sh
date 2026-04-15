#!/bin/bash
# Single H200 job: dense + random-mask sparse BSR-DPO phases (see src/full_training/h200_sparse_dpo_bsr_benchmark.py).
# Submit from repo root:  sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
#
# Northeastern Explorer-style defaults; override env vars as needed.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=h200_bsr_bench
#SBATCH --output=logs/h200_bsr_bench_%j.out
#SBATCH --error=logs/h200_bsr_bench_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Disable W&B / external experiment trackers
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

# --- Model / data (override before sbatch or via #SBATCH --export) ---
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE_ROOT:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_DATASETS_CACHE_ROOT}"

# README-style DPO knobs (100 steps / phase in benchmark script)
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-100}"
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"

OUT_BASE="${H200_BSR_OUT:-${SCRATCH_USER_ROOT}/rl_casino_h200_bsr}/${RUN_ID:-${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_BASE"

echo "REPO_ROOT=${REPO_ROOT}"
echo "OUT_BASE=${OUT_BASE}"
echo "MODEL=${MODEL}"

exec "$TRAIN_PY" src/full_training/h200_sparse_dpo_bsr_benchmark.py \
  --model_name "$MODEL" \
  --dataset tulu3 \
  --n_steps "$NUM_STEPS_DPO" \
  --batch_size "$DPO_PER_DEVICE_TRAIN_BATCH_SIZE" \
  --grad_accum "$DPO_GRADIENT_ACCUMULATION_STEPS" \
  --lr "$DPO_LEARNING_RATE" \
  --warmup_ratio "$DPO_WARMUP_RATIO" \
  --max_length "$DPO_MAX_LENGTH" \
  --max_prompt_length "$DPO_MAX_PROMPT_LENGTH" \
  --output_dir "$OUT_BASE" \
  --dataset_cache_dir "$HF_DATASETS_CACHE" \
  --device_map none \
  --run_label "h200_bsr_${SLURM_JOB_ID:-local}"

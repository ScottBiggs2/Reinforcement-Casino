#!/usr/bin/env bash
# Micro-benchmark: measure wall time for N dense GRPO steps (same hyperparams on every node type).
# Does not submit Slurm; run on an allocated GPU session or wrap with sbatch.
#
# Usage (after: salloc ... or on login with GPU):
#   SCRATCH_USER_ROOT=/scratch/$USER ./scripts/benchmark_grpo_step_time.sh
#
# Env:
#   MODEL, GRPO_BENCH_STEPS (default 20), GRPO_DATASET, HF_DATASETS_CACHE,
#   TRAIN_ENV, RL_CASINO_SCRATCH_ROOT, WANDB_MODE=disabled to skip W&B
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export GRPO_BENCH_STEPS="${GRPO_BENCH_STEPS:-20}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_CONSOLE=off
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"

BENCH_ROOT="${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/benchmarks/step_time_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_ROOT"

echo "=== GRPO step-time benchmark ==="
echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL=${MODEL} STEPS=${GRPO_BENCH_STEPS} OUT=${BENCH_ROOT}"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

/usr/bin/time -p "${TRAIN_PY}" src/full_training/GRPO_train.py \
  --model_name "${MODEL}" \
  --dataset "${GRPO_DATASET}" \
  --num_steps "${GRPO_BENCH_STEPS}" \
  --subset_size 256 \
  --output_base_dir "${BENCH_ROOT}" \
  --dataset_cache_dir "${HF_DATASETS_CACHE}" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_generations 8 \
  --generation_batch_size 8 \
  --save_steps 999999 \
  --optim adamw_torch \
  --run_slug "bench_${GRPO_BENCH_STEPS}steps"

echo "Done. Divide real time by GRPO_BENCH_STEPS for seconds/step."

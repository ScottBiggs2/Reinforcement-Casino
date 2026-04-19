#!/usr/bin/env bash
# Smoke test: sparse GRPO for a few steps with frequent saves, then resume with --resume_from_checkpoint auto.
# Requires one GPU, mask file, and conda env. Run from repo root.
#
# After a successful verify_grpo_training.sh (default model + math-220k), the mask is usually:
#   masks/verify_google_gemma_3_270m_it_math_220k_grpo_dense_step10.pt
#
# Usage:
#   On a GPU node (recommended):
#     salloc -p gpu --gres=gpu:1 -t 0:45:00 --mem=128G ...
#     cd ~/rl_casino && bash scripts/sparse_grpo_resume_smoke.sh
#   Or batch:
#     sbatch scripts/sparse_grpo_resume_smoke_slurm.sh
#
#   MASK_PATH=/path/to/mask.pt bash scripts/sparse_grpo_resume_smoke.sh
#
# If the process is "Killed" with no Python traceback, that is usually OOM (CPU RAM or GPU VRAM).
# Bash prints the line where the `python ...` command *starts* (this file ~79–98), not the line inside Python
# where memory spiked — so "line 98" can mean OOM during the first GRPO generation step, not SparseAdamW.
# With num_generations=1 you used to get a Python ValueError at GRPOConfig *before* trainer.train(); the optimizer
# had already finished — that did not prove the generation phase fit in memory. num_generations>=2 runs that phase.
# Login nodes: use salloc/sbatch. On a GPU node: try larger --mem (e.g. 128G), or lower SMOKE_MAX_COMPLETION_LENGTH.
# GRPO requires num_generations >= 2 (TRL); do not set SMOKE_NUM_GENERATIONS below 2.
# --sparse_adamw_lazy_state avoids eager Adam state pre-allocation (OOM on small GPUs during init only).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]] && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "WARNING: No SLURM_JOB_ID and CUDA_VISIBLE_DEVICES is unset." >&2
  echo "  Running Gemma + GRPO on a login node often gets OOM-killed (shows as: Killed)." >&2
  echo "  Use: salloc -p gpu --gres=gpu:1 -t 0:45:00 --mem=128G bash -l" >&2
  echo "  Then: cd $REPO_ROOT && bash scripts/sparse_grpo_resume_smoke.sh" >&2
  echo "  Or:  sbatch scripts/sparse_grpo_resume_smoke_slurm.sh" >&2
fi

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_CONSOLE=off

MODEL="${MODEL:-google/gemma-3-270m-it}"
# Default: mask produced by scripts/verify_grpo_training.sh (10 steps, gemma + math-220k)
if [[ -z "${MASK_PATH:-}" ]]; then
  MASK_PATH="${REPO_ROOT}/masks/verify_google_gemma_3_270m_it_math_220k_grpo_dense_step10.pt"
fi
if [[ ! -f "$MASK_PATH" ]]; then
  echo "ERROR: mask not found: $MASK_PATH" >&2
  echo "Run scripts/verify_grpo_training.sh first, or set MASK_PATH to any compatible .pt mask." >&2
  exit 1
fi

OUT="${SPARSE_SMOKE_OUT:-${RL_CASINO_SCRATCH_ROOT:-${SCRATCH_USER_ROOT}}/rl_casino_grpo/sparse_resume_smoke_$$}"
RUN_NAME="sparse_resume_smoke_$$"

# First GRPO step runs generation + backward; small GPUs or tight cgroup RAM can OOM-kill the process.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# TRL GRPOConfig enforces num_generations >= 2 (advantages need multiple samples per prompt).
SMOKE_NUM_GENERATIONS="${SMOKE_NUM_GENERATIONS:-2}"
# TRL also requires generation_batch_size % num_generations == 0 (e.g. 2/2, not 1/2).
SMOKE_GENERATION_BATCH_SIZE="${SMOKE_GENERATION_BATCH_SIZE:-${SMOKE_NUM_GENERATIONS}}"
# Short completions shrink GRPO generation memory (KV + activations) more than lowering num_generations (min 2).
SMOKE_MAX_COMPLETION_LENGTH="${SMOKE_MAX_COMPLETION_LENGTH:-256}"
SMOKE_SUBSET_SIZE="${SMOKE_SUBSET_SIZE:-8}"
SMOKE_PRECISION="${SMOKE_PRECISION:-auto}"
if (( SMOKE_NUM_GENERATIONS < 2 )); then
  echo "ERROR: GRPO requires num_generations >= 2 (TRL). Got SMOKE_NUM_GENERATIONS=${SMOKE_NUM_GENERATIONS}." >&2
  exit 1
fi
if (( SMOKE_GENERATION_BATCH_SIZE % SMOKE_NUM_GENERATIONS != 0 )); then
  echo "ERROR: TRL requires generation_batch_size divisible by num_generations. Got gen_bs=${SMOKE_GENERATION_BATCH_SIZE} num_gen=${SMOKE_NUM_GENERATIONS}." >&2
  exit 1
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "MASK_PATH=${MASK_PATH}"
echo "OUT=${OUT} RUN_NAME=${RUN_NAME}"

echo "Phase 1: 2 steps, save every step"
echo "Smoke knobs: num_gen=${SMOKE_NUM_GENERATIONS} gen_bs=${SMOKE_GENERATION_BATCH_SIZE} max_len=${SMOKE_MAX_COMPLETION_LENGTH} subset=${SMOKE_SUBSET_SIZE} precision=${SMOKE_PRECISION}"
"${TRAIN_PY}" src/full_training/sparse_grpo_bsr.py \
  --model_name "${MODEL}" \
  --checkpoint "${MODEL}" \
  --mask "${MASK_PATH}" \
  --n_steps 2 \
  --save_steps 1 \
  --save_total_limit 5 \
  --batch_size 1 \
  --grad_accum 1 \
  --num_generations "${SMOKE_NUM_GENERATIONS}" \
  --generation_batch_size "${SMOKE_GENERATION_BATCH_SIZE}" \
  --max_completion_length "${SMOKE_MAX_COMPLETION_LENGTH}" \
  --precision "${SMOKE_PRECISION}" \
  --dataset math-220k \
  --subset_size "${SMOKE_SUBSET_SIZE}" \
  --output_base_dir "${OUT}" \
  --run_name "${RUN_NAME}" \
  --save_model false \
  --optimizer sparse_adamw \
  --sparse_adamw_lazy_state

echo "Phase 2: resume auto, total steps 4 (continues from step 2)"
"${TRAIN_PY}" src/full_training/sparse_grpo_bsr.py \
  --model_name "${MODEL}" \
  --checkpoint "${MODEL}" \
  --mask "${MASK_PATH}" \
  --n_steps 4 \
  --save_steps 1 \
  --save_total_limit 5 \
  --batch_size 1 \
  --grad_accum 1 \
  --num_generations "${SMOKE_NUM_GENERATIONS}" \
  --generation_batch_size "${SMOKE_GENERATION_BATCH_SIZE}" \
  --max_completion_length "${SMOKE_MAX_COMPLETION_LENGTH}" \
  --precision "${SMOKE_PRECISION}" \
  --dataset math-220k \
  --subset_size "${SMOKE_SUBSET_SIZE}" \
  --output_base_dir "${OUT}" \
  --run_name "${RUN_NAME}" \
  --resume_from_checkpoint auto \
  --save_model false \
  --optimizer sparse_adamw \
  --sparse_adamw_lazy_state

echo "OK: sparse GRPO resume smoke finished."
echo "Artifacts: ${OUT}/${RUN_NAME}/"

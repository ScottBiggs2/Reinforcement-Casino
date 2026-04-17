#!/usr/bin/env bash
# Smoke test: sparse GRPO for a few steps with frequent saves, then resume with --resume_from_checkpoint auto.
# Requires one GPU, mask file, and conda env. Run from repo root.
#
# After a successful verify_grpo_training.sh (default model + math-220k), the mask is usually:
#   masks/verify_google_gemma_3_270m_it_math_220k_grpo_dense_step10.pt
#
# Usage:
#   cd ~/rl_casino && ./scripts/sparse_grpo_resume_smoke.sh
#   MASK_PATH=/path/to/mask.pt ./scripts/sparse_grpo_resume_smoke.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
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

echo "REPO_ROOT=${REPO_ROOT}"
echo "MASK_PATH=${MASK_PATH}"
echo "OUT=${OUT} RUN_NAME=${RUN_NAME}"

echo "Phase 1: 2 steps, save every step"
"${TRAIN_PY}" src/full_training/sparse_grpo_bsr.py \
  --model_name "${MODEL}" \
  --checkpoint "${MODEL}" \
  --mask "${MASK_PATH}" \
  --n_steps 2 \
  --save_steps 1 \
  --save_total_limit 5 \
  --batch_size 1 \
  --grad_accum 1 \
  --num_generations 2 \
  --generation_batch_size 2 \
  --dataset math-220k \
  --subset_size 32 \
  --output_base_dir "${OUT}" \
  --run_name "${RUN_NAME}" \
  --save_model false \
  --optimizer sparse_adamw

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
  --num_generations 2 \
  --generation_batch_size 2 \
  --dataset math-220k \
  --subset_size 32 \
  --output_base_dir "${OUT}" \
  --run_name "${RUN_NAME}" \
  --resume_from_checkpoint auto \
  --save_model false \
  --optimizer sparse_adamw

echo "OK: sparse GRPO resume smoke finished."
echo "Artifacts: ${OUT}/${RUN_NAME}/"

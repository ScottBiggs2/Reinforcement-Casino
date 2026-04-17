#!/usr/bin/env bash
# Smoke test: sparse GRPO for a few steps with frequent saves, then resume with --resume_from_checkpoint auto.
# Requires one GPU, mask file, and conda env. Run from repo root.
#
#   MASK_PATH=masks/foo.pt TRAIN_ENV=... ./scripts/sparse_grpo_resume_smoke.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT:-/scratch/$USER}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_CONSOLE=off

MASK_PATH="${MASK_PATH:?Set MASK_PATH to a .pt mask}"
MODEL="${MODEL:-google/gemma-3-270m-it}"
OUT="${SPARSE_SMOKE_OUT:-${RL_CASINO_SCRATCH_ROOT:-/scratch/$USER}/rl_casino_grpo/sparse_smoke_$$}"
RUN_NAME="sparse_resume_smoke_$$"

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

echo "Phase 2: resume auto, total steps 4 (continues from 2)"
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

echo "OK: sparse GRPO resume smoke finished. Artifacts under ${OUT}/${RUN_NAME}"

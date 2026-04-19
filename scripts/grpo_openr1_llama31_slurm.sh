#!/usr/bin/env bash
# Open-R1 Math GRPO (dense or sparse BSR) with scratch under RL_CASINO_SCRATCH_ROOT.
# Hyperparameter defaults and rationale: docs/hyperparams/open_r1_llama31.yaml
# Runbook: docs/GRPO_OPEN_R1_RUNBOOK.md
#
# Submit from repo root: sbatch scripts/grpo_openr1_llama31_slurm.sh
#
# Examples:
#   GRPO_MODE=dense sbatch scripts/grpo_openr1_llama31_slurm.sh
#   GRPO_MODE=sparse GRPO_MASK=/path/to/mask.pt sbatch scripts/grpo_openr1_llama31_slurm.sh
#   GRPO_RESUME=auto GRPO_TARGET_STEPS=5000 sbatch ...   # same run_slug / run_name as prior job
#
# Explorer / many sites: the gpu partition requires an explicit --gres; omitting it can yield
# "sbatch: error: Batch job submission failed: Access/permission denied". Match other H200 jobs:
#   scripts/h200_sparse_dpo_bsr_benchmark.sh, scripts/pipeline_stage_01_dense.sh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_openr1
#SBATCH --output=logs/grpo_openr1_%j.out
#SBATCH --error=logs/grpo_openr1_%j.err
# If the scheduler rejects gpu:h200:1 (or you need a different GPU type), replace the --gres line above, e.g.:
# #SBATCH --gres=gpu:1

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
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Avoid importing a broken vLLM wheel (see src/utils/trl_vllm_import_guard.py). Set to 0 if you use trl[vllm].
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"

# Fail fast on login/CPU nodes (torch loads fp32 → huge RAM → OOM kill with no Python traceback).
cuda_preflight() {
  "${TRAIN_PY}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print(
        "ERROR: CUDA is not available. Use a GPU allocation (sbatch/salloc with --gres). "
        "On Explorer, `python -c \"import torch; print(torch.cuda.is_available())\"` must print True.",
        file=sys.stderr,
    )
    sys.exit(1)
name = torch.cuda.get_device_name(0)
print(f"CUDA preflight OK: device_count={torch.cuda.device_count()} device0={name!r}")
PY
}

# --- Model / data (override before sbatch) ---
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"
export GRPO_MODE="${GRPO_MODE:-dense}"
export GRPO_NGPUS="${GRPO_NGPUS:-1}"

# Training length / resume
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-1000}"
export GRPO_RESUME="${GRPO_RESUME:-}"  # empty | auto | path to checkpoint-*

# Hyperparams (see docs/hyperparams/open_r1_llama31.yaml)
export GRPO_LR="${GRPO_LR:-5e-6}"
export GRPO_BETA="${GRPO_BETA:-0.025}"
export GRPO_PER_DEVICE_BS="${GRPO_PER_DEVICE_BS:-2}"
export GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-4}"
export GRPO_NUM_GEN="${GRPO_NUM_GEN:-8}"
export GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-8}"
# Passed through to HF Trainer as save_steps / save_total_limit (save_strategy=steps).
# save_steps: write checkpoint-* every N global steps. save_total_limit: keep only the newest K checkpoints on disk (rotation).
# This script does NOT auto-requeue; if wall time stops training early, resubmit with GRPO_RESUME=auto and the same run_slug/run_name.
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"
export GRPO_OPTIM="${GRPO_OPTIM:-adamw_8bit}"
export GRPO_PRECISION="${GRPO_PRECISION:-bf16}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-1024}"
# openr1_tags | llama_cot — see src/utils/grpo_rewards.py (default: delimiter-aware for Instruct models)
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
# 1 = pass --use_wandb to training; 0 = offline / no W&B UI integration
export GRPO_USE_WANDB="${GRPO_USE_WANDB:-1}"
# Sparse only: set 1 to pass --sparse_adamw_lazy_state (lower peak VRAM during SparseAdamW init)
export GRPO_SPARSE_ADAMW_LAZY="${GRPO_SPARSE_ADAMW_LAZY:-0}"

# Optional: fixed run folder for resume requeues (must match first job)
export GRPO_RUN_SLUG="${GRPO_RUN_SLUG:-}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"
echo "GRPO_MODE=${GRPO_MODE} GRPO_NGPUS=${GRPO_NGPUS}"
echo "MODEL=${MODEL} DATASET=${GRPO_DATASET} STEPS=${GRPO_TARGET_STEPS} RESUME=${GRPO_RESUME:-<none>}"

cuda_preflight || exit 1

LAUNCHER=( "${TRAIN_PY}" )
if [ "${GRPO_NGPUS}" -gt 1 ]; then
  LAUNCHER=( "${TRAIN_ENV}/bin/torchrun" --standalone --nproc_per_node="${GRPO_NGPUS}" )
fi

RESUME_ARGS=()
if [ -n "${GRPO_RESUME:-}" ]; then
  RESUME_ARGS+=( --resume_from_checkpoint "${GRPO_RESUME}" )
fi

RUN_SLUG_ARGS=()
if [ -n "${GRPO_RUN_SLUG:-}" ]; then
  RUN_SLUG_ARGS+=( --run_slug "${GRPO_RUN_SLUG}" )
fi

RUN_NAME_ARGS=()
if [ -n "${GRPO_RUN_NAME:-}" ]; then
  RUN_NAME_ARGS+=( --run_name "${GRPO_RUN_NAME}" )
fi

WANDB_ARGS=()
if [ "${GRPO_USE_WANDB:-1}" = "1" ]; then
  WANDB_ARGS+=( --use_wandb )
fi

SPARSE_LAZY_ARGS=()
if [ "${GRPO_SPARSE_ADAMW_LAZY:-0}" = "1" ]; then
  SPARSE_LAZY_ARGS+=( --sparse_adamw_lazy_state )
fi

if [ "${GRPO_MODE}" = "dense" ]; then
  "${LAUNCHER[@]}" src/full_training/GRPO_train.py \
    --model_name "${MODEL}" \
    --dataset "${GRPO_DATASET}" \
    --num_steps "${GRPO_TARGET_STEPS}" \
    --learning_rate "${GRPO_LR}" \
    --beta "${GRPO_BETA}" \
    --per_device_train_batch_size "${GRPO_PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRPO_GRAD_ACCUM}" \
    --num_generations "${GRPO_NUM_GEN}" \
    --generation_batch_size "${GRPO_GEN_BATCH}" \
    --save_steps "${GRPO_SAVE_STEPS}" \
    --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT}" \
    --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH}" \
    --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}" \
    --precision "${GRPO_PRECISION}" \
    --optim "${GRPO_OPTIM}" \
    --grpo_reward_profile "${GRPO_REWARD_PROFILE}" \
    --dataset_cache_dir "${HF_DATASETS_CACHE}" \
    "${WANDB_ARGS[@]}" \
    "${RUN_SLUG_ARGS[@]}" \
    "${RUN_NAME_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
elif [ "${GRPO_MODE}" = "sparse" ]; then
  if [ -z "${GRPO_MASK:-}" ] || [ ! -f "${GRPO_MASK}" ]; then
    echo "ERROR: set GRPO_MASK to a .pt mask file for sparse mode." >&2
    exit 1
  fi
  "${LAUNCHER[@]}" src/full_training/sparse_grpo_bsr.py \
    --model_name "${MODEL}" \
    --checkpoint "${MODEL}" \
    --mask "${GRPO_MASK}" \
    --dataset "${GRPO_DATASET}" \
    --n_steps "${GRPO_TARGET_STEPS}" \
    --lr "${GRPO_LR}" \
    --grpo_beta "${GRPO_BETA}" \
    --batch_size "${GRPO_PER_DEVICE_BS}" \
    --grad_accum "${GRPO_GRAD_ACCUM}" \
    --num_generations "${GRPO_NUM_GEN}" \
    --generation_batch_size "${GRPO_GEN_BATCH}" \
    --save_steps "${GRPO_SAVE_STEPS}" \
    --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT}" \
    --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH}" \
    --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}" \
    --precision "${GRPO_PRECISION}" \
    --grpo_reward_profile "${GRPO_REWARD_PROFILE}" \
    --dataset_cache_dir "${HF_DATASETS_CACHE}" \
    "${WANDB_ARGS[@]}" \
    "${RUN_NAME_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${SPARSE_LAZY_ARGS[@]}"
else
  echo "ERROR: GRPO_MODE must be 'dense' or 'sparse'." >&2
  exit 1
fi

echo ""
echo "To resume this run after requeue, re-submit with the same GRPO_RUN_SLUG (dense) or"
echo "GRPO_RUN_NAME (sparse) and: export GRPO_RESUME=auto"
echo "W&B id is stored under the run directory as wandb_run_id.txt"
echo "Hyperparameter record: docs/hyperparams/open_r1_llama31.yaml"

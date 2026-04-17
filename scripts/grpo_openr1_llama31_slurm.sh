#!/usr/bin/env bash
# Open-R1 Math GRPO (dense or sparse BSR) with scratch under RL_CASINO_SCRATCH_ROOT.
# Submit from repo root: sbatch scripts/grpo_openr1_llama31_slurm.sh
#
# Examples:
#   GRPO_MODE=dense sbatch scripts/grpo_openr1_llama31_slurm.sh
#   GRPO_MODE=sparse GRPO_MASK=/path/to/mask.pt sbatch scripts/grpo_openr1_llama31_slurm.sh
#   GRPO_RESUME=auto GRPO_TARGET_STEPS=5000 sbatch ...   # same run_slug / run_name as prior job
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_openr1
#SBATCH --output=logs/grpo_openr1_%j.out
#SBATCH --error=logs/grpo_openr1_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
## Uncomment and set GPU type/count for your cluster, e.g.:
## #SBATCH --gres=gpu:h200:1
## #SBATCH --gres=gpu:v100:4

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
# Avoid importing a broken vLLM wheel (see src/utils/trl_vllm_import_guard.py). Set to 0 if you use trl[vllm].
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"

# --- Model / data (override before sbatch) ---
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"
export GRPO_MODE="${GRPO_MODE:-dense}"
export GRPO_NGPUS="${GRPO_NGPUS:-1}"

# Training length / resume
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-1000}"
export GRPO_RESUME="${GRPO_RESUME:-}"  # empty | auto | path to checkpoint-*

# Hyperparams (aligned with plan defaults)
export GRPO_LR="${GRPO_LR:-5e-6}"
export GRPO_BETA="${GRPO_BETA:-0.025}"
export GRPO_PER_DEVICE_BS="${GRPO_PER_DEVICE_BS:-2}"
export GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-4}"
export GRPO_NUM_GEN="${GRPO_NUM_GEN:-8}"
export GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-8}"
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_OPTIM="${GRPO_OPTIM:-adamw_8bit}"

# Optional: fixed run folder for resume requeues (must match first job)
export GRPO_RUN_SLUG="${GRPO_RUN_SLUG:-}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"
echo "GRPO_MODE=${GRPO_MODE} GRPO_NGPUS=${GRPO_NGPUS}"
echo "MODEL=${MODEL} DATASET=${GRPO_DATASET} STEPS=${GRPO_TARGET_STEPS} RESUME=${GRPO_RESUME:-<none>}"

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
    --optim "${GRPO_OPTIM}" \
    --dataset_cache_dir "${HF_DATASETS_CACHE}" \
    --use_wandb \
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
    --dataset_cache_dir "${HF_DATASETS_CACHE}" \
    --use_wandb \
    "${RUN_NAME_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
else
  echo "ERROR: GRPO_MODE must be 'dense' or 'sparse'." >&2
  exit 1
fi

echo ""
echo "To resume this run after requeue, re-submit with the same GRPO_RUN_SLUG (dense) or"
echo "GRPO_RUN_NAME (sparse) and: export GRPO_RESUME=auto"
echo "W&B id is stored under the run directory as wandb_run_id.txt"

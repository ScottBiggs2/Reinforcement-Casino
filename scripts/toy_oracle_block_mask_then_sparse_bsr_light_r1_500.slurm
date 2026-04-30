#!/usr/bin/env bash
#
# 1-hour GPU drop-in:
#   1) Build an oracle *block-structured* mask from HF base vs a local checkpoint (|final-initial|).
#   2) Run a 500-step sparse DPO BSR phase on Light-R1 with:
#        - SparseAdamW using the 2D block kernel (RL_CASINO_ADAM_KERNEL=block_2d)
#        - Triton sparse grad_input (RL_CASINO_BSR_GRAD_INPUT_MODE=sparse)
#
# Submit from repo root:
#   sbatch scripts/toy_oracle_block_mask_then_sparse_bsr_light_r1_500.slurm
#
# Key overrides:
#   sbatch --export=ALL,HF_TOKEN=...,TRAIN_ENV=/scratch/$USER/conda_envs/rl_casino scripts/toy_oracle_block_mask_then_sparse_bsr_light_r1_500.slurm
#   sbatch --export=ALL,CKPT500_DIR=/scratch/.../checkpoint-500 scripts/toy_oracle_block_mask_then_sparse_bsr_light_r1_500.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=toy_oracle_blk_bsr
#SBATCH --output=logs/toy_oracle_blk_bsr_%j.out
#SBATCH --error=logs/toy_oracle_blk_bsr_%j.err

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

export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

# W&B: default OFF for toy runs (Slurm/NFS console capture can be flaky).
# Enable by exporting USE_WANDB=1 and (if needed) WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT.
USE_WANDB="${USE_WANDB:-0}"
if [ "${USE_WANDB}" != "1" ]; then
  export WANDB_MODE="disabled"
  export WANDB_DISABLED="true"
  export WANDB_CONSOLE="off"
  export WANDB_SILENT="true"
else
  unset WANDB_DISABLED || true
  export WANDB_CONSOLE="off"
  export WANDB_SILENT="true"
fi

# BSR training knobs requested: 2D optimizer blocks + sparse grad_input.
export RL_CASINO_ADAM_KERNEL="${RL_CASINO_ADAM_KERNEL:-block_2d}"
export RL_CASINO_BSR_GRAD_INPUT_MODE="${RL_CASINO_BSR_GRAD_INPUT_MODE:-sparse}"
export RL_CASINO_BSR_QUIET_INJECTION="${RL_CASINO_BSR_QUIET_INJECTION:-1}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CKPT500_DIR="${CKPT500_DIR:-/scratch/biggs.s/rl_casino_train/dpo5k_dense_light-r1/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500}"

SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
ORACLE_BLOCK_SIZE="${ORACLE_BLOCK_SIZE:-16}"
ORACLE_BLOCK_REDUCTION="${ORACLE_BLOCK_REDUCTION:-mean}" # mean|max

OUT_MASK_DIR="${OUT_MASK_DIR:-${SCRATCH_USER_ROOT}/rl_casino_masks/toy_oracle_block_ckpt_diff/${SLURM_JOB_ID:-local}}"
OUT_TRAIN_DIR="${OUT_TRAIN_DIR:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train/toy_oracle_block_bsr/${SLURM_JOB_ID:-local}}"
mkdir -p "${OUT_MASK_DIR}" "${OUT_TRAIN_DIR}"

MASK_PATH="${OUT_MASK_DIR}/oracle_ckptdiff_block_b${ORACLE_BLOCK_SIZE}_${ORACLE_BLOCK_REDUCTION}_sp${SPARSITY_PERCENT}pct.pt"

echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "CKPT500_DIR=${CKPT500_DIR}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT} MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}"
echo "ORACLE_BLOCK_SIZE=${ORACLE_BLOCK_SIZE} ORACLE_BLOCK_REDUCTION=${ORACLE_BLOCK_REDUCTION}"
echo "MASK_PATH=${MASK_PATH}"
echo "OUT_TRAIN_DIR=${OUT_TRAIN_DIR}"
echo "RL_CASINO_ADAM_KERNEL=${RL_CASINO_ADAM_KERNEL} RL_CASINO_BSR_GRAD_INPUT_MODE=${RL_CASINO_BSR_GRAD_INPUT_MODE}"

"$TRAIN_PY" - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo ""
echo "=== 1) Build oracle BLOCK mask (|final - initial|) ==="
"$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
  --initial_model "${BASE_MODEL}" \
  --final_model "${CKPT500_DIR}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --mask-granularity block \
  --mask-block-size "${ORACLE_BLOCK_SIZE}" \
  --mask-block-reduction "${ORACLE_BLOCK_REDUCTION}" \
  --output_file "${MASK_PATH}"

echo ""
echo "=== 2) Sparse DPO BSR (Light-R1) for 500 steps ==="

# Training defaults (match your Light-R1 settings unless overridden).
N_STEPS="${N_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
LR="${LR:-5e-7}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"

BSR_BLOCK_SIZE="${BSR_BLOCK_SIZE:-16}"
ADAM_BLOCK_SIZE="${ADAM_BLOCK_SIZE:-128}"

RUN_NAME="${RUN_NAME:-toy_oracle_block_bsr_light_r1_ckpt500_${SLURM_JOB_ID:-local}}"

timeout --signal=TERM --kill-after=60 3500 \
  "$TRAIN_PY" src/full_training/sparse_dpo_bsr.py \
    --model_name "${BASE_MODEL}" \
    --checkpoint "${CKPT500_DIR}" \
    --mask "${MASK_PATH}" \
    --dataset light-r1 \
    --n_steps "${N_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --optimizer sparse_adamw \
    --block_size_bsr "${BSR_BLOCK_SIZE}" \
    --block_size_adam "${ADAM_BLOCK_SIZE}" \
    $( [ "${USE_WANDB}" = "1" ] && echo "--use_wandb" ) \
    --save_csv \
    --output_base_dir "${OUT_TRAIN_DIR}" \
    --dataset_cache_dir "${HF_DATASETS_CACHE}" \
    --run_name "${RUN_NAME}"

echo ""
echo "Done."
echo "Mask dir:  ${OUT_MASK_DIR}"
echo "Train dir: ${OUT_TRAIN_DIR}"
ls -lh "${OUT_MASK_DIR}" || true


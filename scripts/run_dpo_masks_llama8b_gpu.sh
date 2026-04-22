#!/bin/bash
# Sparse DPO training on 1× h200 (gpu partition, 8h limit).
# Single-GPU variant of run_dpo_masks_llama8b.sh to bypass multigpu backlog.
#
# Usage:
#   MASK_PATH=/path/to/mask.pt RUN_TAG=oracle_dpo sbatch scripts/run_dpo_masks_llama8b_gpu.sh
#
#SBATCH --job-name=llama8b_sparse_dpo_gpu
#SBATCH --output=logs/llama8b_sparse_dpo_gpu_%j.out
#SBATCH --error=logs/llama8b_sparse_dpo_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=200G
#SBATCH --time=07:45:00

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-tulu3}"
N_STEPS="${N_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
DPO_BETA="${DPO_BETA:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"

export HF_HOME="/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/$USER/hf_cache/datasets"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
COMMON_OUTPUT_DIR="/scratch/$USER/rl_casino_outputs"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$COMMON_OUTPUT_DIR"
if [ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/$USER/.conda/envs/rl_casino || conda activate /scratch/$USER/conda_envs/rl_casino

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=$WANDB_PROJECT

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

GRAD_ACCUM=8
LAUNCH="python"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="llama8b_sparse_dpo_${RUN_TAG:-unknown}_1gpu_${TIMESTAMP}"
OUTPUT_DIR="${COMMON_OUTPUT_DIR}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "1-GPU Sparse DPO — $RUN_NAME"
echo "Mask: ${MASK_PATH:-NONE}"
echo "N_STEPS=$N_STEPS  LR=$LR  DPO_BETA=$DPO_BETA  MAX_GRAD_NORM=$MAX_GRAD_NORM"
echo "=================================================="

if [ -z "${MASK_PATH:-}" ] || [ ! -f "$MASK_PATH" ]; then
    echo "ERROR: MASK_PATH must be set and exist. Got: ${MASK_PATH:-}" >&2
    exit 2
fi

$LAUNCH src/full_training/sparse_dpo_bsr.py \
    --model_name "$MODEL" \
    --checkpoint None \
    --mask "$MASK_PATH" \
    --dataset "$DATASET" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --dpo_beta "$DPO_BETA" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --optimizer sparse_adamw \
    --block_size_bsr 16 \
    --block_size_adam 128 \
    --use_wandb \
    --save_csv \
    --save_model true \
    --output_base_dir "$OUTPUT_DIR" \
    --dataset_cache_dir "$HF_DATASETS_CACHE" \
    --run_name "$RUN_NAME" 2>&1

STATUS=$?
echo "=================================================="
echo "Finished: $(date) — status=$STATUS"
[ $STATUS -eq 0 ] && echo "SUCCESS: $RUN_NAME" || echo "FAILED: $RUN_NAME ($STATUS)"
echo "=================================================="
exit $STATUS

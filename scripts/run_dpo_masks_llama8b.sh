#!/bin/bash
# Sparse DPO training with Llama-3.1-8B-Instruct on multi-GPU.
# Mirrors run_dpo_masks_grpo_llama8b.sh but for DPO (light-r1 dataset).
#
# Usage:
#   MASK_PATH=/path/to/mask.pt RUN_TAG=oracle_dpo sbatch scripts/run_dpo_masks_llama8b.sh
#   MASK_PATH=/path/to/cav.pt  RUN_TAG=cav_dpo    sbatch scripts/run_dpo_masks_llama8b.sh
#
#SBATCH --job-name=llama8b_sparse_dpo
#SBATCH --output=logs/llama8b_sparse_dpo_%j.out
#SBATCH --error=logs/llama8b_sparse_dpo_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=07:45:00

set -euo pipefail

# === CONFIGURATION === (env-overridable for monitor auto-intervention)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-tulu3}"
N_STEPS="${N_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
DPO_BETA="${DPO_BETA:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"

# === STORAGE ===
export HF_HOME="/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/$USER/hf_cache/datasets"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
COMMON_OUTPUT_DIR="/scratch/$USER/rl_casino_outputs"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$COMMON_OUTPUT_DIR"

if [ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

# === ENV ===
source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/$USER/.conda/envs/rl_casino || conda activate /scratch/$USER/conda_envs/rl_casino

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export WANDB_PROJECT=$WANDB_PROJECT

# === REPO ROOT ===
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# === Multi-GPU ===
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NGPUS="$SLURM_GPUS_ON_NODE"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NGPUS=1
fi
GRAD_ACCUM=$((8 / NGPUS))
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

if [ "$NGPUS" -gt 1 ]; then
    LAUNCH="accelerate launch --num_processes $NGPUS --multi_gpu"
else
    LAUNCH="python"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="llama8b_sparse_dpo_${RUN_TAG:-unknown}_${NGPUS}gpu_${TIMESTAMP}"
OUTPUT_DIR="${COMMON_OUTPUT_DIR}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Job started: $(date)  |  Node: $(hostname)  |  Job: ${SLURM_JOB_ID:-local}"
echo "GPUs: $NGPUS | Grad accum: $GRAD_ACCUM"
echo "Mask: ${MASK_PATH:-NONE}"
echo "Run name: $RUN_NAME"
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
echo ""
echo "=================================================="
echo "Job finished: $(date)"
if [ $STATUS -eq 0 ]; then
    echo "SUCCESS: ${RUN_NAME}"
    echo "Output dir: ${OUTPUT_DIR}"
else
    echo "FAILED: ${RUN_NAME} (exit ${STATUS})"
fi
echo "=================================================="
exit $STATUS

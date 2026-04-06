#!/bin/bash
# Run GRPO training (sparse or dense) with Llama-3.1-8B-Instruct on a single GPU.
# Usage:
#   MASK_PATH=/path/to/mask.pt RUN_TAG=cav_dpo sbatch scripts/run_dpo_masks_grpo_llama8b.sh
#   RUN_TAG=dense_baseline sbatch scripts/run_dpo_masks_grpo_llama8b.sh   # no mask = dense
#SBATCH --job-name=llama8b_grpo
#SBATCH --output=logs/llama8b_grpo_%j.out
#SBATCH --error=logs/llama8b_grpo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# === CONFIGURATION ===
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="math-220k"
N_STEPS=200
SUBSET=512
BATCH_SIZE=1
GRAD_ACCUM=8
NUM_GENERATIONS=8
GEN_BATCH_SIZE=8
LR=1e-6
WANDB_PROJECT="huggingface"

# === STORAGE (all on /scratch) ===
export HF_HOME="/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/$USER/hf_cache/datasets"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
COMMON_OUTPUT_DIR="/scratch/$USER/rl_casino_outputs"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$COMMON_OUTPUT_DIR"

# Copy HF token to scratch HF_HOME if it exists in default location
if [ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

# === ENVIRONMENT SETUP ===
source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/$USER/.conda/envs/rl_casino || conda activate /scratch/$USER/conda_envs/rl_casino

export PYTHONPATH="$(pwd):$PYTHONPATH"
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

echo "=================================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Working dir: $(pwd)"
echo "HF_HOME: $HF_HOME"
echo "Output dir: $COMMON_OUTPUT_DIR"
echo "Project: $WANDB_PROJECT"
echo "=================================================="

# === TIMESTAMP & RUN NAME ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="llama8b_dpo_mask_grpo_${RUN_TAG:-unknown}_${TIMESTAMP}"
OUTPUT_DIR="${COMMON_OUTPUT_DIR}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# === RUN ===
if [ -n "${MASK_PATH:-}" ] && [ -f "$MASK_PATH" ]; then
    echo "Sparse GRPO with mask: ${MASK_PATH}"
    echo "Run name: ${RUN_NAME}"
    echo "=================================================="

    python src/full_training/sparse_grpo_bsr.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --n_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --lr "$LR" \
        --optimizer sparse_adamw \
        --mask "$MASK_PATH" \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$HF_DATASETS_CACHE" \
        --use_wandb \
        --save_model true \
        --run_name "$RUN_NAME" 2>&1
else
    echo "Dense GRPO Baseline (no mask)"
    echo "Run name: ${RUN_NAME}"
    echo "=================================================="

    python src/full_training/GRPO_train.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --num_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$HF_DATASETS_CACHE" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --use_wandb \
        --run_name "$RUN_NAME" 2>&1
fi

STATUS=$?
echo ""
echo "=================================================="
echo "Job finished at: $(date)"
if [ $STATUS -eq 0 ]; then
    echo "SUCCESS: ${RUN_NAME}"
else
    echo "FAILED: ${RUN_NAME} (exit code $STATUS)"
fi
echo "=================================================="
exit $STATUS

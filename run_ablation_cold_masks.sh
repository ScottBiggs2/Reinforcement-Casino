#!/bin/bash
#SBATCH --job-name=ablation_cold
#SBATCH --output=logs/ablation_cold_%j.out
#SBATCH --error=logs/ablation_cold_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu

set -e

# Configuration
MODEL="google/gemma-3-270m-it"
SUBSET=256
STEPS=100
BATCH=4
GRAD_ACCUM=4
LR=5e-5

# Source conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
conda activate /scratch/biggs.s/conda_envs/rl_casino
export PYTHONPATH=.

echo "Starting Cold-Start Mask Ablation..."
echo "Comparing Dense Backprop vs Sparse Backprop (BSR)"

run_mask_ablation() {
    local MASK_PATH=$1
    local MASK_NAME=$2
    
    if [ ! -f "$MASK_PATH" ]; then
        echo "Warning: Mask not found at $MASK_PATH, skipping..."
        return
    fi

    echo "----------------------------------------------------"
    echo "Ablating Mask: $MASK_NAME"
    echo "----------------------------------------------------"

    # 1. Dense Backprop (Baseline)
    echo "Running Dense Backprop Baseline..."
    python3 src/full_training/sparse_dpo_efficiency.py \
        --model_name "$MODEL" \
        --mask "$MASK_PATH" \
        --n_steps $STEPS \
        --batch_size $BATCH \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --subset_size $SUBSET \
        --optimizer "sparse_adamw" \
        --use_wandb \
        --run_name "ablation_${MASK_NAME}_dense_bp"

    # 2. Sparse Backprop (BSR)
    echo "Running Sparse Backprop (BSR)..."
    python3 src/full_training/sparse_dpo_bsr.py \
        --model_name "$MODEL" \
        --mask "$MASK_PATH" \
        --n_steps $STEPS \
        --batch_size $BATCH \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --subset_size $SUBSET \
        --optimizer "sparse_adamw" \
        --use_wandb \
        --run_name "ablation_${MASK_NAME}_sparse_bp"
}

# --- Cold Start Masks ---
run_mask_ablation "masks/cold_fisher_google_gemma_3_270m_it_qihoo360_Light-R1-DPOData_sparsity97.5pct_n256.pt" "cold_fisher"
run_mask_ablation "masks/cold_cav_google_gemma_3_270m_it_sparsity97.5pct.pt" "cold_cav"

echo "Cold-Start Ablation Complete!"

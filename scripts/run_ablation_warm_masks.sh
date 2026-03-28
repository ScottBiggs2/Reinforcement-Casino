#!/bin/bash
#SBATCH --job-name=ablation_warm
#SBATCH --output=logs/ablation_warm_%j.out
#SBATCH --error=logs/ablation_warm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu

# Source conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
conda activate /scratch/biggs.s/conda_envs/rl_casino

# Set up PYTHONPATH
export PYTHONPATH=.

# Configuration
MODEL="google/gemma-3-270m-it"
SUBSET=256
STEPS=100
BATCH=4
GRAD_ACCUM=4
LR=5e-5
TARGET_STEP=50
SPARSITY=97.5
LOG_DIR="delta_logs_google_gemma_3_270m_it"
MIN_LAYER_KEEP_RATIO="0.0025" # set to 0.0 for pure global masking

echo "Generating Warm-Start Masks..."

echo "-> Magnitude Mask"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method magnitude \
  --sparsity_percent $SPARSITY \
  --target_step $TARGET_STEP \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard 

echo "-> Momentum Mask"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method momentum \
  --sparsity_percent $SPARSITY \
  --target_step $TARGET_STEP \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard 

echo "-> Fisher Mask (Warm)"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method fisher \
  --sparsity_percent $SPARSITY \
  --target_step $TARGET_STEP \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard 

echo "Starting Warm-Start Mask Ablation..."
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

# --- Warm Start Masks ---
run_mask_ablation "masks/warm_fisher_google_gemma_3_270m_it_sparsity97.5pct_step50.pt" "warm_fisher_step50"
run_mask_ablation "masks/warm_magnitude_google_gemma_3_270m_it_sparsity97.5pct_step50.pt" "warm_magnitude_step50"
run_mask_ablation "masks/warm_momentum_w5_google_gemma_3_270m_it_sparsity97.5pct_step50.pt" "warm_momentum_step50"

echo "Warm-Start Ablation Complete!"

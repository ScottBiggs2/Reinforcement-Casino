#!/bin/bash
#SBATCH --job-name=ablation_baseline
#SBATCH --output=logs/ablation_baseline_%j.out
#SBATCH --error=logs/ablation_baseline_%j.err
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
SPARSITY=97.5
LOG_DIR="delta_logs_google_gemma_3_270m_it"

echo "Starting Random & Ground Truth Mask Ablation..."

# --- 1. Baseline DPO Training (Self-Contained Kickoff) ---
echo "=================================================="
echo "1. Running Baseline DPO Training"
echo "=================================================="
python src/full_training/DPO_train.py --model_name "$MODEL" --use_wandb --run_name "ablation_baseline_dpo"

# --- 2. Calculate GT Mask ---
echo "=================================================="
echo "2. Calculating GT Mask"
echo "=================================================="
echo "-> Ground Truth Mask"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method magnitude \
  --sparsity_percent $SPARSITY \
  --target_step 500 \
  --mlp_only \
  --compute_jaccard 

REF_MASK_GT="masks/warm_magnitude_google_gemma_3_270m_it_sparsity${SPARSITY}pct_step500.pt"

# --- 3. Generate Truly Random Mask ---
echo "=================================================="
echo "3. Generating Random Mask"
echo "=================================================="
RANDOM_MASK="masks/random_sample_sparsity${SPARSITY}pct.pt"

echo "Generating random mask using reference mask for shape matching..."
python3 src/warm_start/random_mask_baseline.py \
    --reference_mask "$REF_MASK_GT" \
    --sparsity_percent "$SPARSITY" \
    --seed 42 \
    --compare_to_reference \
    --output_file "$RANDOM_MASK"

echo ""
echo "Comparing BSR-AdamW + Dense vs Sparse Backprop"

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
    echo "Running BSR-AdamW + Dense Backprop Baseline..."
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
    echo "Running BSR-AdamW + Sparse Backprop (BSR)..."
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

# --- Baselines ---
run_mask_ablation "$RANDOM_MASK" "random_baseline"
run_mask_ablation "$REF_MASK_GT" "ground_truth_magnitude"

echo "Random & GT Ablation Complete!"

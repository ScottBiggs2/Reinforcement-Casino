#!/bin/bash
# Warm-Start Mask Ablation: mini DPO training run with each warm-start mask.
# Tests 3 conditions per mask: dense AdamW, BSR sparse backprop + AdamW, BSR sparse backprop + SparseAdamW.
# WandB only — no checkpoints saved to spare HPC disk space.
# Run from project dir: sbatch run_ablation_warm_masks.sh
#SBATCH --job-name=warm_mask_ablation
#SBATCH --output=logs/warm_mask_ablation_%j.out
#SBATCH --error=logs/warm_mask_ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# =============================================================
# Environment
# =============================================================
source ~/miniconda3/etc/profile.d/conda.sh || \
  source ~/anaconda3/etc/profile.d/conda.sh || \
  source /opt/conda/etc/profile.d/conda.sh
conda activate /scratch/biggs.s/conda_envs/rl_casino
export PYTHONPATH=.

# =============================================================
# Shared hyperparameters
# =============================================================
MODEL="google/gemma-3-270m-it"
N_STEPS=100
LR=5e-6
BATCH=2
GRAD_ACCUM=4          # effective batch = 8
DPO_BETA=0.1
MAX_GRAD_NORM=1.0
BLOCK_BSR=16
BLOCK_ADAM=128

mkdir -p logs

run_trio() {
    # run_trio <mask_path> <label>
    #   Executes three training runs for one mask:
    #     1. BSR sparse back + dense AdamW  (mask controls backprop kernel only)
    #     2. BSR sparse back + SparseAdamW  (mask controls backprop AND weight update)
    #     3. Random baseline paired with this mask (for SNR reference)
    local MASK=$1
    local LABEL=$2

    echo ""
    echo "============================================================"
    echo "MASK: $LABEL"
    echo "============================================================"

    # --- BSR Sparse backprop + standard AdamW ---
    echo "  -> BSR Sparse backprop + AdamW"
    python src/full_training/sparse_dpo_bsr.py \
        --model_name "$MODEL" \
        --mask "$MASK" \
        --n_steps $N_STEPS \
        --batch_size $BATCH \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --dpo_beta $DPO_BETA \
        --max_grad_norm $MAX_GRAD_NORM \
        --block_size_bsr $BLOCK_BSR \
        --block_size_adam $BLOCK_ADAM \
        --optimizer adamw \
        --mlp_only \
        --use_wandb \
        --run_name "ablation_warm_${LABEL}_bsr_adamw"

    # --- BSR Sparse backprop + SparseAdamW ---
    echo "  -> BSR Sparse backprop + SparseAdamW"
    python src/full_training/sparse_dpo_bsr.py \
        --model_name "$MODEL" \
        --mask "$MASK" \
        --n_steps $N_STEPS \
        --batch_size $BATCH \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --dpo_beta $DPO_BETA \
        --max_grad_norm $MAX_GRAD_NORM \
        --block_size_bsr $BLOCK_BSR \
        --block_size_adam $BLOCK_ADAM \
        --optimizer sparse_adamw \
        --mlp_only \
        --use_wandb \
        --run_name "ablation_warm_${LABEL}_bsr_sparse_adamw"
}

# =============================================================
# Warm-start masks (step=50 variants — early checkpoint)
# =============================================================
run_trio \
  "masks/warm_magnitude_google_gemma_3_270m_it_sparsity97.5pct_step50.pt" \
  "magnitude_step50"

run_trio \
  "masks/warm_momentum_w5_google_gemma_3_270m_it_sparsity97.5pct_step50.pt" \
  "momentum_step50"

run_trio \
  "masks/warm_fisher_google_gemma_3_270m_it_sparsity97.5pct_step50.pt" \
  "fisher_step50"

# =============================================================
# Ground truth (converged magnitude mask — upper bound reference)
# =============================================================
run_trio \
  "masks/warm_magnitude_google_gemma_3_270m_it_sparsity97.5pct.pt" \
  "magnitude_gt"

# =============================================================
# Random baselines paired with the warm masks (for SNR reference)
# =============================================================
run_trio \
  "masks/random_baseline_vs_magnitude_sparsity97.5pct.pt" \
  "random_vs_magnitude"

run_trio \
  "masks/random_baseline_vs_momentum_sparsity97.5pct.pt" \
  "random_vs_momentum"

run_trio \
  "masks/random_baseline_vs_fisher_warm_sparsity97.5pct.pt" \
  "random_vs_fisher_warm"

run_trio \
  "masks/random_baseline_vs_ground_truth_sparsity97.5pct.pt" \
  "random_vs_gt"

echo ""
echo "=================================================="
echo "Warm-start mask ablation complete!"
echo "=================================================="

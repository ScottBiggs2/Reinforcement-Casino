#!/bin/bash
# Cold-Start Mask Ablation: mini DPO training run with each cold-start mask.
# Tests 3 conditions per mask: dense AdamW, BSR sparse backprop + AdamW, BSR sparse backprop + SparseAdamW.
# WandB only — no checkpoints saved to spare HPC disk space.
# Run from project dir: sbatch run_ablation_cold_masks.sh
#SBATCH --job-name=cold_mask_ablation
#SBATCH --output=logs/cold_mask_ablation_%j.out
#SBATCH --error=logs/cold_mask_ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

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
    #     1. Dense backprop   + standard AdamW
    #     2. BSR sparse back  + standard AdamW
    #     3. BSR sparse back  + SparseAdamW
    local MASK=$1
    local LABEL=$2

    echo ""
    echo "============================================================"
    echo "MASK: $LABEL"
    echo "============================================================"

    # --- Dense AdamW (no sparse backprop, no mask applied) ---
    # This run has already been done, so it's been commented out for this job routine
    # echo "  -> Dense AdamW"
    # python src/full_training/DPO_train.py \
    #     --model_name "$MODEL" \
    #     2>/dev/null &
    # # DPO_train.py reports to wandb natively with its own run name.
    # # We kill it after N_STEPS via max_steps in DPOConfig (already hardcoded as 500).
    # # Instead call sparse_dpo_bsr with adamw optimizer, which respects --n_steps cleanly.
    # wait

    # Use sparse_dpo_bsr.py with adamw (no actual sparsity injected in dense mode is not how
    # sparse_dpo_bsr works — it always injects BSR layers). For a clean dense baseline we use
    # DPO_train.py but limit steps by pointing to the same underlying config. 
    # Better: use sparse_dpo_bsr --optimizer adamw (injects BSR mask for backprop shape,
    # but adamw still updates ALL mask positions — this is the "dense-optimizer, sparse-kernel" condition).
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
        --run_name "ablation_cold_${LABEL}_dense_adamw"

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
        --run_name "ablation_cold_${LABEL}_bsr_adamw"

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
        --run_name "ablation_cold_${LABEL}_bsr_sparse_adamw"
}

# =============================================================
# Cold-start masks
# =============================================================
run_trio \
  "masks/cold_fisher_google_gemma_3_270m_it_qihoo360_Light-R1-DPOData_sparsity97.5pct_n256.pt" \
  "fisher"

run_trio \
  "masks/cold_cav_google_gemma_3_270m_it_sparsity97.5pct.pt" \
  "cav"

# Random baselines paired with the cold masks (same sparsity pattern, random selection)
run_trio \
  "masks/random_baseline_vs_fisher_cold_sparsity97.5pct.pt" \
  "random_vs_fisher_cold"

run_trio \
  "masks/random_baseline_vs_cav_cold_sparsity97.5pct.pt" \
  "random_vs_cav_cold"

echo ""
echo "=================================================="
echo "Cold-start mask ablation complete!"
echo "=================================================="

#!/bin/bash
# Random Baseline & Ground Truth Ablation: one BSR+SparseAdamW run per random mask, 100 steps.
# These serve as the noise floor / SNR reference — if a mask trained with a random selection
# performs similarly to a principled mask, the mask signal is weak.
# WandB logging, no checkpoints.
# Run from project dir: sbatch run_ablation_random_gt.sh
#SBATCH --job-name=random_gt_ablation
#SBATCH --output=logs/random_gt_ablation_%j.out
#SBATCH --error=logs/random_gt_ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

source ~/miniconda3/etc/profile.d/conda.sh || \
  source ~/anaconda3/etc/profile.d/conda.sh || \
  source /opt/conda/etc/profile.d/conda.sh
conda activate /scratch/biggs.s/conda_envs/rl_casino
export PYTHONPATH=.
mkdir -p logs

MODEL="google/gemma-3-270m-it"
N_STEPS=100
LR=5e-6
BATCH=2
GRAD_ACCUM=4

run_mask() {
    local MASK=$1
    local LABEL=$2
    echo ""
    echo "===> $LABEL"
    python src/full_training/sparse_dpo_bsr.py \
        --model_name "$MODEL" \
        --mask "$MASK" \
        --n_steps $N_STEPS \
        --batch_size $BATCH \
        --grad_accum $GRAD_ACCUM \
        --lr $LR \
        --dpo_beta 0.1 \
        --max_grad_norm 1.0 \
        --block_size_bsr 16 \
        --block_size_adam 128 \
        --optimizer sparse_adamw \
        --mlp_only \
        --use_wandb \
        --run_name "ablation_random_${LABEL}"
}

# Random baselines matched to each mask category
# (same sparsity + structure as the real masks, but random parameter selection)
run_mask \
  "masks/random_baseline_vs_magnitude_sparsity97.5pct.pt" \
  "vs_magnitude"

run_mask \
  "masks/random_baseline_vs_momentum_sparsity97.5pct.pt" \
  "vs_momentum"

run_mask \
  "masks/random_baseline_vs_fisher_warm_sparsity97.5pct.pt" \
  "vs_fisher_warm"

run_mask \
  "masks/random_baseline_vs_ground_truth_sparsity97.5pct.pt" \
  "vs_ground_truth"

run_mask \
  "masks/random_baseline_vs_fisher_cold_sparsity97.5pct.pt" \
  "vs_fisher_cold"

run_mask \
  "masks/random_baseline_vs_cav_cold_sparsity97.5pct.pt" \
  "vs_cav_cold"

echo ""
echo "Random baseline + ground truth ablation complete!"

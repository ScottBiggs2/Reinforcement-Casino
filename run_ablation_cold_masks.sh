#!/bin/bash
# Cold-Start Mask Ablation: one BSR+SparseAdamW run per cold mask, 100 steps.
# Early sanity check only — WandB logging, no checkpoints.
# Run from project dir: sbatch run_ablation_cold_masks.sh
#SBATCH --job-name=cold_mask_ablation
#SBATCH --output=logs/cold_mask_ablation_%j.out
#SBATCH --error=logs/cold_mask_ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

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
        --run_name "ablation_cold_${LABEL}"
}

run_mask \
  "masks/cold_fisher_google_gemma_3_270m_it_qihoo360_Light-R1-DPOData_sparsity97.5pct_n256.pt" \
  "fisher"

run_mask \
  "masks/cold_cav_google_gemma_3_270m_it_sparsity97.5pct.pt" \
  "cav"

echo ""
echo "Cold-start ablation complete!"

#!/bin/bash
# Sparse GRPO arm for the matched-schedule GRPO set on AICR. Two instantiations:
#
#   MASK_PATH=$MASK_DIR/oracle_grpo_math220k_step500_sp97.5.pt  RUN_TAG=oracle_grpo_math
#   MASK_PATH=$MASK_DIR/random_baseline_lightr1_sp97.5_seed42.pt RUN_TAG=random_seed42
#
# Schedule-matched to aicr_grpo_matched_dense.sbatch: cosine over 500 steps, 50-step
# warmup, lr 5e-6, cap 2048, and — critically — max_grad_norm 0.1 now reaches
# GRPOConfig (fix of 2026-07-27; before that, sparse arms silently ran TRL's global
# clip at the HF default 1.0 while dense clipped at 0.1). A gap between this arm and
# dense is therefore attributable to sparsity; a gap between the two sparse arms is
# attributable to the mask — the random control that never existed for GRPO.
#
# random_baseline_lightr1_sp97.5_seed42.pt is the paper's own random baseline; its
# per-layer keep profile comes from the Light-R1 context (disclose as "the paper's
# random baseline mask", not "density-matched to the GRPO oracle") — see
# docs/run_logs/grpo_random_control_2026-07-26.md §3.
#
# Usage: MASK_PATH=... RUN_TAG=... sbatch aicr_grpo_matched_sparse.sbatch
#SBATCH --job-name=grpo_sparse
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/grpo_sparse_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

: "${MASK_PATH:?Set MASK_PATH=/scratch/.../*.pt}"
: "${RUN_TAG:?Set RUN_TAG=oracle_grpo_math or random_seed42}"
[ -f "$MASK_PATH" ] || { echo "FATAL: MASK_PATH not found: $MASK_PATH"; exit 1; }

export WANDB_PROJECT="rl_casino_rebuttal"
OUT_BASE="$S/rebuttal_analysis/grpo_matched_aicr/${RUN_TAG}"
mkdir -p "$OUT_BASE"

echo "=========================================="
echo "Matched-schedule GRPO — SPARSE ($RUN_TAG) | rho=97.5% | AICR b200"
echo "  steps=500 cosine warmup=50 lr=5e-6 beta=0.025 clip=0.1 cap=2048"
echo "  mask=$MASK_PATH"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

python src/full_training/sparse_grpo_bsr.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --mask "$MASK_PATH" \
    --dataset math-220k \
    --n_steps 500 \
    --lr 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --grpo_beta 0.025 \
    --max_grad_norm 0.1 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_prompt_length 512 \
    --max_completion_length 2048 \
    --grpo_reward_profile llama_cot \
    --optimizer sparse_adamw \
    --save_steps 50 \
    --save_total_limit 3 \
    --save_model false \
    --output_base_dir "$OUT_BASE" \
    --use_wandb \
    --run_name "grpo_matched_aicr_${RUN_TAG}_500steps_cap2048" \
    --resume_from_checkpoint auto

echo "Finished: $(date)"

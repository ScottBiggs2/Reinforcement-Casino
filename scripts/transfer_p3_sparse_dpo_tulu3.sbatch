#!/bin/bash
# Parameterized sparse DPO run on Tulu3 with a given mask.
# Usage: MASK_PATH=<path> RUN_TAG=<tag> sbatch transfer_p3_sparse_dpo_tulu3.sbatch
#SBATCH --job-name=transfer_p3_sparse_dpo_tulu3
#SBATCH --output=logs/transfer_p3_sparse_dpo_tulu3_%j.out
#SBATCH --error=logs/transfer_p3_sparse_dpo_tulu3_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export HF_DATASETS_CACHE="/scratch/xie.yiyi/hf_cache/datasets"
export WANDB_PROJECT="rl_casino_transfer_v1"

# Required env vars
: "${MASK_PATH:?Set MASK_PATH=/scratch/.../oracle_*.pt}"
: "${RUN_TAG:?Set RUN_TAG=oracle_dpo_tulu3 | oracle_dpo_lightr1 | oracle_grpo_math | random}"

OUT_BASE="/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_${RUN_TAG}"
mkdir -p "$OUT_BASE"

if [ ! -f "$MASK_PATH" ]; then
  echo "FATAL: MASK_PATH not found: $MASK_PATH"
  exit 1
fi

echo "=========================================="
echo "Phase 3: Sparse DPO  |  Llama-3.1-8B-Instruct  |  Tulu3"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "RUN_TAG   = $RUN_TAG"
echo "MASK_PATH = $MASK_PATH"
echo "OUT_BASE  = $OUT_BASE"
echo "Hyperparams (matching dense Phase 1A, Scott spec):"
echo "  steps=500, eff_batch=128 (per_device 2 × grad_accum 64 × 1 GPU)"
echo "  lr=5e-7, beta=0.1, warmup_ratio=0.1, max_length=1024"
echo "  optimizer=sparse_adamw (mask-aware, doesn't decay frozen weights)"
echo "=========================================="

python src/full_training/sparse_dpo_efficiency.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset tulu3 \
    --mask "$MASK_PATH" \
    --n_steps 500 \
    --batch_size 2 \
    --grad_accum 64 \
    --lr 5e-7 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --dpo_beta 0.1 \
    --optimizer sparse_adamw \
    --save_steps 50 \
    --gradient_checkpointing \
    --output_base_dir "$OUT_BASE" \
    --run_name "sparse_dpo_tulu3_${RUN_TAG}_500steps" \
    --use_wandb \
    --resume_from_checkpoint auto

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -10
echo "=========================================="

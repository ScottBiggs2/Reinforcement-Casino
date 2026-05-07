#!/bin/bash
# Parameterized sparse GRPO run on math-220k with a given mask.
# Usage: MASK_PATH=<path> RUN_TAG=<tag> sbatch transfer_p3_sparse_grpo_math220k.sbatch
#SBATCH --job-name=transfer_p3_sparse_grpo
#SBATCH --output=logs/transfer_p3_sparse_grpo_%j.out
#SBATCH --error=logs/transfer_p3_sparse_grpo_%j.err
#SBATCH --partition=gpu
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

: "${MASK_PATH:?Set MASK_PATH=/scratch/.../oracle_*.pt}"
: "${RUN_TAG:?Set RUN_TAG=oracle_dpo_lightr1_500 | oracle_dpo_lightr1_150 | oracle_dpo_tulu3 | oracle_grpo_math | random}"

OUT_BASE="/scratch/xie.yiyi/transfer_v1/sparse_grpo_math220k_${RUN_TAG}"
mkdir -p "$OUT_BASE"

if [ ! -f "$MASK_PATH" ]; then
  echo "FATAL: MASK_PATH not found: $MASK_PATH"
  exit 1
fi

echo "=========================================="
echo "Phase 3: Sparse GRPO  |  Llama-3.1-8B-Instruct (cold) + mask  |  math-220k"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "RUN_TAG   = $RUN_TAG"
echo "MASK_PATH = $MASK_PATH"
echo "OUT_BASE  = $OUT_BASE"
echo "Hyperparams (matching dense Phase 1C):"
echo "  steps=500, per_device_batch=8, num_generations=8, generation_batch=8"
echo "  lr=5e-6, grpo_beta=0.025, warmup_ratio=0.1"
echo "  optimizer=sparse_adamw + BSR sparse-MLP"
echo "=========================================="

python src/full_training/sparse_grpo_bsr.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --mask "$MASK_PATH" \
    --n_steps 500 \
    --batch_size 2 \
    --grad_accum 4 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --lr 5e-6 \
    --grpo_beta 0.025 \
    --warmup_steps 50 \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --optimizer sparse_adamw \
    --save_steps 50 \
    --save_total_limit 8 \
    --output_base_dir "$OUT_BASE" \
    --run_name "sparse_grpo_math220k_${RUN_TAG}_500steps" \
    --use_wandb \
    --resume_from_checkpoint auto

echo ""
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -10

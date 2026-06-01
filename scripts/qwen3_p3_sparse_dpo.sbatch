#!/bin/bash
# Phase 3 (Qwen3): Sparse DPO on Qwen/Qwen3-8B with a 97.5% mask + mask-aware
# SparseAdamW (does NOT decay frozen weights). This is the high-sparsity DPO
# treatment. Mirrors transfer_p3_sparse_dpo_* hparams exactly.
# Usage: DATASET=light-r1|tulu3 MASK_PATH=<...pt> RUN_TAG=oracle_step500 sbatch qwen3_p3_sparse_dpo.sbatch
#SBATCH --job-name=qwen3_p3_sparse_dpo
#SBATCH --output=logs/qwen3_p3_sparse_dpo_%j.out
#SBATCH --error=logs/qwen3_p3_sparse_dpo_%j.err
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

MODEL_BASE="Qwen/Qwen3-8B"
DATASET="${DATASET:?Set DATASET=light-r1 | tulu3}"
DS_TAG=$(echo "$DATASET" | tr '-' '_')
: "${MASK_PATH:?Set MASK_PATH=/scratch/.../oracle_dpo_${DS_TAG}_step500_sp97.5.pt}"
: "${RUN_TAG:?Set RUN_TAG=oracle_step500 | oracle_step150 | random}"

OUT_BASE="/scratch/xie.yiyi/transfer_v1/sparse_dpo_${DS_TAG}_qwen3_8b_${RUN_TAG}"
mkdir -p "$OUT_BASE"

if [ ! -f "$MASK_PATH" ]; then echo "FATAL: MASK_PATH not found: $MASK_PATH"; exit 1; fi

echo "=========================================="
echo "Phase 3 (Qwen3): Sparse DPO (high-sparsity treatment)  |  $MODEL_BASE  |  $DATASET"
echo "Job ID : ${SLURM_JOB_ID:-local}   Started: $(date)"
echo "RUN_TAG   = $RUN_TAG"
echo "MASK_PATH = $MASK_PATH"
echo "OUT_BASE  = $OUT_BASE"
echo "Hparams: 500 steps, eff_batch 128 (per_device 2 x grad_accum 64 x 1 GPU)"
echo "         lr=5e-7, beta=0.1, warmup_ratio=0.1, max_length=1024, optimizer=sparse_adamw"
echo "=========================================="

python src/full_training/sparse_dpo_efficiency.py \
    --model_name "$MODEL_BASE" \
    --dataset "$DATASET" \
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
    --run_name "sparse_dpo_qwen3_8b_${DS_TAG}_${RUN_TAG}_500steps" \
    --use_wandb \
    --resume_from_checkpoint auto

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -10
echo "=========================================="

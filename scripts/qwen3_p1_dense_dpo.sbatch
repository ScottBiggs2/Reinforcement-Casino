#!/bin/bash
# Phase 1 (Qwen3): Dense DPO on Qwen/Qwen3-8B (instruct/post-trained, NOT -Base).
# Mirrors transfer_p1_dense_dpo_*_8b hyperparams exactly; only the backbone and
# output paths change. Saves checkpoint-150 + checkpoint-500 as oracle sources.
# Usage: DATASET=light-r1|tulu3 sbatch qwen3_p1_dense_dpo.sbatch
#SBATCH --job-name=qwen3_p1_dense_dpo
#SBATCH --output=logs/qwen3_p1_dense_dpo_%j.out
#SBATCH --error=logs/qwen3_p1_dense_dpo_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
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

MODEL_BASE="Qwen/Qwen3-8B"          # instruct/post-trained chat model (Qwen3-8B-Base is the base — do NOT use)
DATASET="${DATASET:?Set DATASET=light-r1 | tulu3}"
DS_TAG=$(echo "$DATASET" | tr '-' '_')   # light-r1 -> light_r1 (matches sanitize_model_name dataset suffix)

OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_dpo_${DS_TAG}_qwen3_8b"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)
echo "=========================================="
echo "Phase 1 (Qwen3): Dense DPO  |  $MODEL_BASE  |  $DATASET"
echo "Job ID : ${SLURM_JOB_ID:-local}   Visible GPUs: $NGPUS   Started: $(date)"
echo "Hparams: 500 steps, eff_batch 128 (per_device 2 x grad_accum 64 x ${NGPUS} GPU)"
echo "         lr=5e-7, beta=0.1, warmup_ratio=0.1, max_length=1024"
echo "         save_steps=50, save_total_limit=12 (retains ckpt-150 + ckpt-500 for oracle)"
echo "OUT_BASE = $OUT_BASE"
echo "=========================================="

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/DPO_train.py \
    --model_name "$MODEL_BASE" \
    --dataset "$DATASET" \
    --num_steps 500 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --save_steps 50 \
    --save_total_limit 12 \
    --gradient_checkpointing \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_name "dense_dpo_qwen3_8b_${DS_TAG}_scott_full"

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -5
find "$OUT_BASE" -maxdepth 4 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -12
echo "=========================================="

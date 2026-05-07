#!/bin/bash
#SBATCH --job-name=transfer_p2c_oracle_grpo_step150
#SBATCH --output=logs/transfer_p2c_oracle_grpo_step150_%j.out
#SBATCH --error=logs/transfer_p2c_oracle_grpo_step150_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
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

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
SPARSITY=97.5
MIN_LAYER_KEEP_RATIO=0.0025
OUT_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
mkdir -p "$OUT_DIR"

# Look for step-150 ckpt under GRPO dense output dir
DENSE_OUT="/scratch/xie.yiyi/transfer_v1/dense_grpo_math220k_llama8b"
CKPT=$(find "$DENSE_OUT" -maxdepth 5 -type d -name "checkpoint-150" 2>/dev/null | head -1)
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
  echo "FATAL: no checkpoint-150 found under $DENSE_OUT"
  echo "Available checkpoints:"
  find "$DENSE_OUT" -maxdepth 5 -type d -name "checkpoint-*" 2>/dev/null | sort -V
  exit 1
fi

OUT="$OUT_DIR/oracle_grpo_math220k_step150_sp${SPARSITY}.pt"

echo "=========================================="
echo "Phase 2C-extra: Oracle (step 150, early) GRPO math-220k"
echo "  for Mukherjee fig 5/6 'subnet emerges early' persistence check"
echo "Started: $(date)"
echo "CKPT = $CKPT"
echo "OUT  = $OUT"
echo "=========================================="

python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT"

echo ""
echo "Finished: $(date)"
ls -lh "$OUT"

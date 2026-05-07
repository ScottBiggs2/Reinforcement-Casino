#!/bin/bash
#SBATCH --job-name=transfer_p2b_oracle_dpo_tulu3
#SBATCH --output=logs/transfer_p2b_oracle_dpo_tulu3_%j.out
#SBATCH --error=logs/transfer_p2b_oracle_dpo_tulu3_%j.err
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

# Auto-detect latest ckpt under Tulu3 dense output dir (final step should be 500)
DENSE_OUT="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b"
CKPT=$(find "$DENSE_OUT" -maxdepth 4 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
  echo "FATAL: no checkpoint found under $DENSE_OUT"
  exit 1
fi

STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
OUT="$OUT_DIR/oracle_dpo_tulu3_step${STEP}_sp${SPARSITY}.pt"

echo "=========================================="
echo "Phase 2B: Oracle (ckpt-diff) DPO Tulu3"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "BASE = $MODEL_BASE"
echo "CKPT = $CKPT (step $STEP)"
echo "OUT  = $OUT"
echo "=========================================="

python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT"

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -lh "$OUT"
echo "=========================================="

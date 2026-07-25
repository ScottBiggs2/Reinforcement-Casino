#!/bin/bash
# Phase 2 (Qwen3-32B): Oracle (checkpoint-diff) mask @97.5% from the dense 32B run.
# CPU-bound: loads initial + final 32B state dicts and diffs them → needs lots of RAM.
# Usage: DATASET=light-r1 sbatch qwen3_32b_p2_oracle_dpo.sbatch
#SBATCH --job-name=qwen3_32b_p2_oracle
#SBATCH --output=logs/qwen3_32b_p2_oracle_%j.out
#SBATCH --error=logs/qwen3_32b_p2_oracle_%j.err
#SBATCH --partition=short
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --requeue
# CPU-only (no --gres): checkpoint-diff is pure CPU tensor work; running it on a
# GPU partition would leave the GPU idle and trip RC's idle auto-cancel (§7b).

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"

MODEL_BASE="Qwen/Qwen3-32B"
SPARSITY=97.5
MIN_LAYER_KEEP_RATIO=0.0025
DATASET="${DATASET:?Set DATASET=light-r1 | tulu3}"
DS_TAG=$(echo "$DATASET" | tr '-' '_')

DENSE_OUT="/scratch/xie.yiyi/transfer_v1/dense_dpo_${DS_TAG}_qwen3_32b"
OUT_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_qwen3_32b"
mkdir -p "$OUT_DIR"

CKPT_500=$(find "$DENSE_OUT" -maxdepth 4 -type d -name "checkpoint-500" 2>/dev/null | head -1)
CKPT_150=$(find "$DENSE_OUT" -maxdepth 4 -type d -name "checkpoint-150" 2>/dev/null | head -1)
if [ -z "$CKPT_500" ] || [ ! -d "$CKPT_500" ]; then
  CKPT_500=$(find "$DENSE_OUT" -maxdepth 4 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
fi
if [ -z "$CKPT_500" ] || [ ! -d "$CKPT_500" ]; then echo "FATAL: no final ckpt under $DENSE_OUT"; exit 1; fi
STEP_FINAL=$(basename "$CKPT_500" | sed 's/checkpoint-//')
OUT_500="$OUT_DIR/oracle_dpo_${DS_TAG}_step${STEP_FINAL}_sp${SPARSITY}.pt"

echo "=========================================="
echo "Phase 2 (Qwen3-32B): Oracle (ckpt-diff)  |  $DATASET  |  sparsity=${SPARSITY}%"
echo "Job ID : ${SLURM_JOB_ID:-local}   Started: $(date)"
echo "BASE=$MODEL_BASE  CKPT_500=$CKPT_500 (step $STEP_FINAL)  CKPT_150=${CKPT_150:-<skip>}"
echo "OUT_500=$OUT_500"
echo "=========================================="

echo "[1/2] Oracle from step $STEP_FINAL (the high-sparsity treatment mask)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT_500" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --force_cpu \
    --output_file "$OUT_500"

if [ -n "$CKPT_150" ] && [ -d "$CKPT_150" ]; then
  OUT_150="$OUT_DIR/oracle_dpo_${DS_TAG}_step150_sp${SPARSITY}.pt"
  echo "[2/2] Oracle from step 150 (trajectory-style)"
  python src/warm_start/checkpoint_diff_mask_finder.py \
      --initial_model "$MODEL_BASE" \
      --final_model "$CKPT_150" \
      --sparsity_percent "$SPARSITY" \
      --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --force_cpu \
      --output_file "$OUT_150"
fi

echo ""
echo "Finished: $(date)"
ls -lh "$OUT_DIR"/oracle_dpo_${DS_TAG}_*_sp${SPARSITY}.pt 2>&1
echo "=========================================="

# Advance the chain: once the step-500 mask exists, this launches p3 sparse DPO.
# (Mask generation is idempotent, so if this slot is interrupted a resubmit just
#  regenerates it — advance.sh re-detects the missing mask and resubmits p2.)
bash "$PWD/scripts/qwen3_32b_advance.sh" || echo "advance.sh rc=$? (nonzero => STALL halt or done)"

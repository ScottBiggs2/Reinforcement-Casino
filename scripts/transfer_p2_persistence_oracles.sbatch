#!/bin/bash
# After a long dense run, generate {step 150, latest} oracle masks for Mukherjee persistence check.
# Usage: DATASET={tulu3|grpo} sbatch transfer_p2_persistence_oracles.sbatch
#SBATCH --job-name=persistence_oracles
#SBATCH --output=logs/persistence_oracles_%j.out
#SBATCH --error=logs/persistence_oracles_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
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

: "${DATASET:?Set DATASET=tulu3 or grpo}"

case "$DATASET" in
  tulu3)
    DENSE_OUT="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_long_llama8b"
    LABEL="dpo_tulu3_long"
    ;;
  grpo)
    DENSE_OUT="/scratch/xie.yiyi/transfer_v1/dense_grpo_math220k_long_llama8b"
    LABEL="grpo_math220k_long"
    ;;
  *) echo "FATAL: DATASET must be tulu3 or grpo"; exit 1 ;;
esac

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
SPARSITY=97.5
MIN_LAYER_KEEP_RATIO=0.0025
OUT_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
mkdir -p "$OUT_DIR"

# Find step 150 ckpt + latest ckpt
CKPT_150=$(find "$DENSE_OUT" -maxdepth 5 -type d -name "checkpoint-150" 2>/dev/null | head -1)
CKPT_LATE=$(find "$DENSE_OUT" -maxdepth 5 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)

if [ -z "$CKPT_150" ] || [ ! -d "$CKPT_150" ]; then echo "FATAL: missing checkpoint-150 in $DENSE_OUT"; exit 1; fi
if [ -z "$CKPT_LATE" ] || [ ! -d "$CKPT_LATE" ]; then echo "FATAL: missing latest ckpt in $DENSE_OUT"; exit 1; fi

LATE_STEP=$(basename "$CKPT_LATE" | sed "s/checkpoint-//")

OUT_150="$OUT_DIR/oracle_${LABEL}_step150_sp${SPARSITY}.pt"
OUT_LATE="$OUT_DIR/oracle_${LABEL}_step${LATE_STEP}_sp${SPARSITY}.pt"

echo "=========================================="
echo "Persistence oracles for $DATASET"
echo "  CKPT_150  = $CKPT_150"
echo "  CKPT_LATE = $CKPT_LATE (step $LATE_STEP)"
echo "  OUT_150   = $OUT_150"
echo "  OUT_LATE  = $OUT_LATE"
echo "=========================================="

echo ""
echo "[1/2] Oracle from step 150 (early)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT_150" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT_150"

echo ""
echo "[2/2] Oracle from step $LATE_STEP (converged)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT_LATE" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT_LATE"

echo ""
echo "=== Jaccard between them ==="
python -c "
import torch
m150 = torch.load('$OUT_150', map_location='cpu', weights_only=False)
mlate = torch.load('$OUT_LATE', map_location='cpu', weights_only=False)
masks_150 = m150.get('masks', m150)
masks_late = mlate.get('masks', mlate)
inter, union = 0, 0
per_layer = []
for k in set(masks_150) & set(masks_late):
    a = masks_150[k].bool()
    b = masks_late[k].bool()
    i = (a & b).sum().item()
    u = (a | b).sum().item()
    inter += i; union += u
    per_layer.append(i / max(u, 1))
print(f'$LABEL: Jaccard(step150, step$LATE_STEP) = {inter/max(union,1):.4f}')
print(f'  per-layer mean: {sum(per_layer)/len(per_layer):.4f}')
print(f'  per-layer min:  {min(per_layer):.4f}')
print(f'  per-layer max:  {max(per_layer):.4f}')
"

echo ""
ls -lh "$OUT_150" "$OUT_LATE"

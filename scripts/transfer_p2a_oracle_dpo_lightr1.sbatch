#!/bin/bash
#SBATCH --job-name=transfer_p2a_oracle_dpo_lightr1
#SBATCH --output=logs/transfer_p2a_oracle_dpo_lightr1_%j.out
#SBATCH --error=logs/transfer_p2a_oracle_dpo_lightr1_%j.err
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

CKPT_BASE="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1"

# Two ckpts: Scott-style (step 500) + trajectory-style (step 150)
CKPT_500="$CKPT_BASE/checkpoint-500"
CKPT_150="$CKPT_BASE/checkpoint-150"

OUT_500="$OUT_DIR/oracle_dpo_lightr1_step500_sp${SPARSITY}.pt"
OUT_150="$OUT_DIR/oracle_dpo_lightr1_step150_sp${SPARSITY}.pt"

echo "=========================================="
echo "Phase 2A: Oracle (ckpt-diff) DPO Light-R1"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "BASE      = $MODEL_BASE"
echo "CKPT_500  = $CKPT_500"
echo "CKPT_150  = $CKPT_150"
echo "OUT_500   = $OUT_500  (Scott-style: step 500, deep training, ~21 epoch)"
echo "OUT_150   = $OUT_150  (trajectory-style: step 150, just-converged, ~6.3 epoch)"
echo "=========================================="

if [ ! -d "$CKPT_500" ]; then echo "FATAL: missing $CKPT_500"; exit 1; fi
if [ ! -d "$CKPT_150" ]; then echo "FATAL: missing $CKPT_150"; exit 1; fi

echo ""
echo "[1/2] Oracle from step 500 (Scott-style)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT_500" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT_500"

echo ""
echo "[2/2] Oracle from step 150 (trajectory-style)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT_150" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT_150"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo ""
ls -lh "$OUT_500" "$OUT_150" 2>&1
echo ""
echo "=== sanity Jaccard between the two oracles ==="
python -c "
import torch
m500 = torch.load('$OUT_500', map_location='cpu', weights_only=False)
m150 = torch.load('$OUT_150', map_location='cpu', weights_only=False)
masks_500 = m500.get('masks', m500)
masks_150 = m150.get('masks', m150)
inter, union = 0, 0
for k in set(masks_500) & set(masks_150):
    a = masks_500[k].bool()
    b = masks_150[k].bool()
    inter += (a & b).sum().item()
    union += (a | b).sum().item()
jac = inter / max(union, 1)
print(f'Jaccard(oracle_step500, oracle_step150) = {jac:.4f}')
print(f'  (high jaccard = overfit drift only marginally changes the mask set)')
print(f'  (low jaccard  = step 150 vs 500 select very different subnets)')
"
echo "=========================================="

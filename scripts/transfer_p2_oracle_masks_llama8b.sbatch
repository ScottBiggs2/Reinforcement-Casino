#!/bin/bash
#SBATCH --job-name=transfer_p2_oracle_8b
#SBATCH --output=logs/transfer_p2_oracle_8b_%j.out
#SBATCH --error=logs/transfer_p2_oracle_8b_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

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
SPARSITY="${SPARSITY:-97.5}"
OUT_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
mkdir -p "$OUT_DIR"

# DPO/GRPO final checkpoint paths — set by Phase 1 outputs (override if Phase 1 used different schedule).
DPO_CKPT="${DPO_CKPT:-$(ls -d /scratch/xie.yiyi/transfer_v1/dense_dpo_llama8b/checkpoint-* 2>/dev/null | sort -V | tail -1)}"
GRPO_CKPT="${GRPO_CKPT:-$(ls -d /scratch/xie.yiyi/transfer_v1/dense_grpo_llama8b/llama8b_math220k_dense_100steps/checkpoint-* 2>/dev/null | sort -V | tail -1)}"

if [ -z "$DPO_CKPT" ] || [ ! -d "$DPO_CKPT" ]; then echo "FATAL: missing DPO_CKPT (set explicitly)"; exit 1; fi
if [ -z "$GRPO_CKPT" ] || [ ! -d "$GRPO_CKPT" ]; then echo "FATAL: missing GRPO_CKPT (set explicitly)"; exit 1; fi

DPO_OUT="$OUT_DIR/oracle_dpo_llama8b_lightr1_$(basename $DPO_CKPT)_sp${SPARSITY}.pt"
GRPO_OUT="$OUT_DIR/oracle_grpo_llama8b_math220k_$(basename $GRPO_CKPT)_sp${SPARSITY}.pt"

echo "=========================================="
echo "Phase 2: Oracle (ckpt-diff) masks  |  Llama-3.1-8B-Instruct"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "BASE     = $MODEL_BASE"
echo "DPO_CKPT = $DPO_CKPT"
echo "GRPO_CKPT= $GRPO_CKPT"
echo "DPO_OUT  = $DPO_OUT"
echo "GRPO_OUT = $GRPO_OUT"
echo "=========================================="

echo ""
echo "[1/2] Oracle-DPO (Light-R1)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$DPO_CKPT" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio 0.0025 \
    --output_file "$DPO_OUT"

echo ""
echo "[2/2] Oracle-GRPO (math-220k)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$GRPO_CKPT" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio 0.0025 \
    --output_file "$GRPO_OUT"

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -lh "$DPO_OUT" "$GRPO_OUT"
echo "=========================================="

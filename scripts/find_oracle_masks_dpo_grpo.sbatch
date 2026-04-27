#!/bin/bash
#SBATCH --job-name=oracle_masks_dpo_grpo
#SBATCH --output=logs/oracle_masks_%j.out
#SBATCH --error=logs/oracle_masks_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export HF_DATASETS_CACHE="/scratch/xie.yiyi/hf_cache/datasets"

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
SPARSITY="${SPARSITY:-97.5}"
OUT_DIR="${OUT_DIR:-/scratch/xie.yiyi/rl_casino_masks/oracle_v2}"
mkdir -p "$OUT_DIR"

DPO_FINAL="${DPO_FINAL:-/scratch/xie.yiyi/rl_casino_outputs_v2/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500}"
GRPO_FINAL="${GRPO_FINAL:-/scratch/xie.yiyi/rl_casino_outputs/llama8b_dpo_mask_grpo_dense_baseline_1gpu_20260408_044634/checkpoints/meta_llama_llama_3_1_8b_instruct_math_220k_grpo_dense/checkpoint-200}"

DPO_OUT="${OUT_DIR}/oracle_dpo_light_r1_ckpt500_sp${SPARSITY}.pt"
GRPO_OUT="${OUT_DIR}/oracle_grpo_math220k_ckpt200_sp${SPARSITY}.pt"

echo "=========================================="
echo "Oracle (ground-truth ckpt-diff) mask job"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "MODEL_BASE  = $MODEL_BASE"
echo "SPARSITY    = $SPARSITY"
echo "DPO_FINAL   = $DPO_FINAL"
echo "GRPO_FINAL  = $GRPO_FINAL"
echo "DPO_OUT     = $DPO_OUT"
echo "GRPO_OUT    = $GRPO_OUT"
echo "=========================================="

if [ ! -d "$DPO_FINAL" ]; then echo "FATAL: missing DPO_FINAL: $DPO_FINAL"; exit 1; fi
if [ ! -d "$GRPO_FINAL" ]; then echo "FATAL: missing GRPO_FINAL: $GRPO_FINAL"; exit 1; fi

echo ""
echo "[1/2] Oracle-DPO  (Light-R1, step 500)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model   "$DPO_FINAL" \
    --sparsity_percent "$SPARSITY" \
    --output_file "$DPO_OUT"

echo ""
echo "[2/2] Oracle-GRPO (math-220k, step 200)"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model   "$GRPO_FINAL" \
    --sparsity_percent "$SPARSITY" \
    --output_file "$GRPO_OUT"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo ""
ls -lh "$DPO_OUT" "$GRPO_OUT" 2>&1
echo "=========================================="

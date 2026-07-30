#!/bin/bash
# Phase 2 (Qwen3-32B, AICR): oracle (checkpoint-diff) mask @97.5% from the p1
# dense checkpoint-500. Pure CPU tensor work -> `cpu` partition (1.13 TB nodes,
# no idle-GPU canceller on AICR so no keep-alive sidecar needed).
#
# Memory: the 8B checkpoint-diff route peaked ~105 GB at rho=97.5; 32.8B params
# is ~4.1x -> ~430-500 GB expected. 800G requested.
#
# Usage: sbatch scripts/aicr_qwen32b_p2_oracle.sbatch   (or via aicr_qwen32b_chain.sh)
#SBATCH --job-name=q32b_p2_oracle
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/q32b_p2_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

MODEL_BASE="Qwen/Qwen3-32B"
SPARSITY=97.5
MIN_LAYER_KEEP_RATIO=0.0025

DENSE_OUT=$S/transfer_v1/dense_dpo_light_r1_qwen3_32b
OUT_DIR=$S/transfer_v1/oracle_masks_qwen3_32b
mkdir -p "$OUT_DIR"

CKPT_500=$DENSE_OUT/checkpoints/qwen_qwen3_32b_light_r1/checkpoint-500
if [ ! -d "$CKPT_500" ]; then echo "FATAL: $CKPT_500 does not exist — p1 not complete"; exit 1; fi
OUT_500="$OUT_DIR/oracle_dpo_light_r1_step500_sp${SPARSITY}.pt"
if [ -f "$OUT_500" ]; then echo "mask already exists: $OUT_500"; exit 0; fi

echo "=========================================="
echo "Phase 2 (Qwen3-32B @ AICR): oracle ckpt-diff | sparsity ${SPARSITY}%"
echo "Job ${SLURM_JOB_ID:-local} | $(date) | CKPT_500=$CKPT_500"
echo "=========================================="

python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "$MODEL_BASE" \
    --final_model "$CKPT_500" \
    --sparsity_percent "$SPARSITY" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --force_cpu \
    --output_file "$OUT_500"

ls -lh "$OUT_DIR"/oracle_dpo_light_r1_*_sp${SPARSITY}.pt
echo "Finished: $(date)"

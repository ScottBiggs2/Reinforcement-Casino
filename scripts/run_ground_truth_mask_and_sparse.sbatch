#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --job-name=gt_mask_sparse
#SBATCH --output=logs/gt_mask_sparse_%j.out
#SBATCH --error=logs/gt_mask_sparse_%j.err

set -euo pipefail

export PIPELINE_RUN_ID="tulu3_500_h200_fresh_0409_again"
source scripts/pipeline_common.sh
pipeline_setup

MASK_OUT="${MASK_OUT_BASE}/${RUN_ID}/warm_ground_truth_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step${TARGET_STEP_DPO}.pt"
DELTA_DIR="${TRAIN_OUT_BASE}/${RUN_ID}/deltas/meta_llama_llama_3_1_8b_instruct_tulu3"

echo "===== PHASE 1: GROUND TRUTH MASK GENERATION ====="
echo "Delta dir: ${DELTA_DIR}"
echo "Target step: ${TARGET_STEP_DPO}"
echo "Output: ${MASK_OUT}"

python src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "${DELTA_DIR}" \
    --method ground_truth \
    --sparsity_percent 97.5 \
    --target_step "${TARGET_STEP_DPO}" \
    --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
    --output_file "${MASK_OUT}"

echo "Mask generated at: ${MASK_OUT}"
ls -lh "${MASK_OUT}"

echo "===== PHASE 2: SPARSE DPO TRAINING ====="
run_sparse_dpo_one_mask "${MASK_OUT}"

echo "===== DONE ====="

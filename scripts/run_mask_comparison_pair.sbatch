#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask_cmp
#SBATCH --output=logs/mask_cmp_%j.out
#SBATCH --error=logs/mask_cmp_%j.err

# Exhaustive pairwise mask comparison: Jaccard + per-layer CSV + plots.
# Usage: sbatch --export=ALL,MASK_A=/path/a.pt,MASK_B=/path/b.pt scripts/run_mask_comparison_pair.sbatch
# Or override COMP_TAG to customise output filenames (default: auto-generated from basenames).
#
# Outputs land in ${MASK_OUT_BASE}/${RUN_ID}/comparisons/  (same as main pipeline)
# unless COMP_OUT_DIR is set explicitly.

set -euo pipefail

export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-tulu3_500_h200_fresh_0409_again}"
source scripts/pipeline_common.sh
pipeline_setup

if [ -z "${MASK_A:-}" ] || [ -z "${MASK_B:-}" ]; then
    echo "ERROR: MASK_A and MASK_B must be set." >&2
    echo "  sbatch --export=ALL,MASK_A=/path/to/a.pt,MASK_B=/path/to/b.pt $0" >&2
    exit 1
fi

if [ ! -f "${MASK_A}" ]; then
    echo "ERROR: MASK_A not found: ${MASK_A}" >&2
    exit 1
fi
if [ ! -f "${MASK_B}" ]; then
    echo "ERROR: MASK_B not found: ${MASK_B}" >&2
    exit 1
fi

BASE_A=$(basename "${MASK_A}" .pt)
BASE_B=$(basename "${MASK_B}" .pt)
COMP_TAG="${COMP_TAG:-${BASE_A}_vs_${BASE_B}}"

COMP_OUT_DIR="${COMP_OUT_DIR:-${MASK_OUT_BASE}/${RUN_ID}/comparisons}"
PLOT_DIR="${COMP_OUT_DIR}/plots"
mkdir -p "${COMP_OUT_DIR}" "${PLOT_DIR}"

JACCARD_JSON="${COMP_OUT_DIR}/jaccard_${COMP_TAG}.json"
CKA_JSON="${COMP_OUT_DIR}/cka_${COMP_TAG}.json"
LAYER_CSV="${COMP_OUT_DIR}/layer_metrics_${COMP_TAG}.csv"

echo "===== MASK COMPARISON ====="
echo "  A: ${MASK_A}"
echo "  B: ${MASK_B}"
echo "  Tag: ${COMP_TAG}"
echo "  Out: ${COMP_OUT_DIR}"

echo "--- Step 1: Jaccard similarity ---"
python src/cold_start/mask_to_jaccard.py \
    "${MASK_A}" "${MASK_B}" \
    -o "${JACCARD_JSON}"

if [ "${RUN_MASK_CKA:-0}" = "1" ]; then
    echo "--- Step 2: CKA similarity ---"
    python src/cold_start/mask_to_cka.py \
        "${MASK_A}" "${MASK_B}" \
        --model_name "${MODEL}" \
        --dataset_name "${COLD_DATASET_HF}" \
        --device cpu \
        --n_samples "${CKA_N_SAMPLES:-64}" \
        --batch_size "${CKA_BATCH_SIZE:-4}" \
        --seed 42 \
        -o "${CKA_JSON}"
else
    echo "--- Step 2: CKA skipped (RUN_MASK_CKA not set to 1) ---"
fi

echo "--- Step 3: Per-layer CSV export ---"
EXPORT_CMD=(
    python src/cold_start/export_layer_metrics_csv.py
    "${MASK_A}" "${MASK_B}"
    --jaccard-json "${JACCARD_JSON}"
    -o "${LAYER_CSV}"
    --skip_effective_rank
)
if [ -f "${CKA_JSON}" ]; then
    EXPORT_CMD+=(--cka-json "${CKA_JSON}")
fi
"${EXPORT_CMD[@]}"

echo "--- Step 4: Plot generation ---"
python src/cold_start/plot_layer_metrics_csv.py \
    --input-dir "${COMP_OUT_DIR}" \
    --recursive \
    --pattern "layer_metrics_${COMP_TAG}.csv" \
    --output-dir "${PLOT_DIR}" \
    --random-trials "${PLOT_RANDOM_TRIALS:-3}"

echo "===== COMPARISON COMPLETE ====="
echo "Results:"
echo "  Jaccard JSON : ${JACCARD_JSON}"
echo "  Layer CSV    : ${LAYER_CSV}"
echo "  Plots        : ${PLOT_DIR}"

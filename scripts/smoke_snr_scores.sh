#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test for optional SNR score reweighting.
# Runs small model + tiny calibration; writes three masks and prints aggregate Jaccard.
#
# Example:
#   bash scripts/smoke_snr_scores.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
MODEL="${MODEL:-google/gemma-2-2b-it}"
OUT_DIR="${OUT_DIR:-masks/smoke_snr}"
mkdir -p "$OUT_DIR"

echo "PY=$PY"
echo "MODEL=$MODEL"
echo "OUT_DIR=$OUT_DIR"

BASE="${OUT_DIR}/snip_pref_base.pt"
SNR="${OUT_DIR}/snip_pref_snr_per_tensor.pt"

echo ""
echo "=== SNIP dpo_preference baseline ==="
$PY src/cold_start/inference_mask_finder.py \
  --model_name "$MODEL" \
  --method snip \
  --mode dpo \
  --dataset_name tulu3 \
  --snip-objective dpo_preference \
  --n_samples 16 \
  --sparsity 90.0 \
  --batch_size 1 \
  --max_length 256 \
  --snip-num-batches 4 \
  --output "$BASE"

echo ""
echo "=== SNIP dpo_preference + SNR(per_tensor) ==="
$PY src/cold_start/inference_mask_finder.py \
  --model_name "$MODEL" \
  --method snip \
  --mode dpo \
  --dataset_name tulu3 \
  --snip-objective dpo_preference \
  --n_samples 16 \
  --sparsity 90.0 \
  --batch_size 1 \
  --max_length 256 \
  --snip-num-batches 4 \
  --score-snr per_tensor \
  --score-snr-transform log1p \
  --output "$SNR"

echo ""
echo "=== Jaccard comparison (baseline vs SNR) ==="
$PY src/cold_start/inference_mask_finder.py \
  --model_name "$MODEL" \
  --method snip \
  --mode dpo \
  --dataset_name tulu3 \
  --snip-objective dpo_preference \
  --n_samples 16 \
  --sparsity 90.0 \
  --batch_size 1 \
  --max_length 256 \
  --snip-num-batches 1 \
  --reference_mask "$BASE" \
  --output "${OUT_DIR}/_tmp_refcheck.pt" >/dev/null

echo "Done."


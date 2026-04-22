#!/usr/bin/env bash
set -euo pipefail

# Smoke test for CAV v2 (all-params) mask generator + coverage verifier.
#
# Intended to run locally on a GPU machine, but also works on CPU with small models.
#
# Example:
#   bash scripts/smoke_test_cav_v2.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL="${MODEL:-google/gemma-3-270m-it}"
MODE="${MODE:-dpo}"
SPARSITY="${SPARSITY_PERCENT:-97.5}"
N="${N_SAMPLES:-8}"
BS="${BATCH_SIZE:-2}"
MAXLEN="${MAX_LENGTH:-128}"
MIN_KEEP="${MIN_LAYER_KEEP_RATIO:-0.0}"
WEIGHT_ABS="${WEIGHT_ABS:-1}"

OUT="masks/_smoke_cav_v2_${MODE}_$(echo "$MODEL" | tr '/-' '__')_sp${SPARSITY}.pt"
REPORT="${OUT%.pt}_coverage.json"

abs_args=()
if [ "$WEIGHT_ABS" = "1" ]; then abs_args+=( --weight_abs ); fi

echo "=== CAV v2 smoke test ==="
echo "MODEL=$MODEL MODE=$MODE SPARSITY=$SPARSITY N=$N BS=$BS MAXLEN=$MAXLEN MIN_KEEP=$MIN_KEEP WEIGHT_ABS=$WEIGHT_ABS"

$PY src/cold_start/cav_v2_all_params_mask_finder.py \
  --model_name "$MODEL" \
  --mode "$MODE" \
  --n_samples "$N" \
  --batch_size "$BS" \
  --max_length "$MAXLEN" \
  --sparsity_percent "$SPARSITY" \
  --min_layer_keep_ratio "$MIN_KEEP" \
  --output_file "$OUT" \
  --coverage_report_out "$REPORT" \
  "${abs_args[@]}"

echo ""
echo "=== Coverage verify (hard gate) ==="
$PY src/utils/verify_mask_coverage.py \
  --model_name "$MODEL" \
  --mask_file "$OUT" \
  --out_json "$REPORT"

echo "✓ Smoke test passed: $OUT"


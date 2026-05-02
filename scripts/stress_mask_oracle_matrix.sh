#!/usr/bin/env bash
# Stress-test mask interpretation: pairwise Jaccard among oracle/GT masks and vs random baselines.
#
# Usage (from repo root):
#   ORACLE_MASKS="task_a/gt.pt task_b/gt.pt task_c/gt.pt" \
#   RANDOM_BASELINE_DIR=/path/to/random \
#   bash scripts/stress_mask_oracle_matrix.sh
#
# - ORACLE_MASKS: whitespace-separated list of absolute or repo-relative .pt paths.
# - OUT_DIR: output directory (default: ./mask_stress_oracle_matrix_out)
# - SPARSITY_FOR_RANDOM: passed to warm_start/random_mask_baseline.py if RANDOM_BASELINE_DIR
#   is set but no random .pt exists (optional; requires REFERENCE_MASK for topology).
#
# Pass criteria (printed at end; non-zero exit if any fail):
#   - Every oracle-oracle pair with shared keys has aggregate Jaccard > ORACLE_ORACLE_MIN (default 0.01)
#   - Every oracle-random pair has aggregate Jaccard < ORACLE_RANDOM_MAX (default 0.35)
#     (tune for your sparsity; at ~90% sparsity random-vs-random E[J] is ~0.053)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/mask_stress_oracle_matrix_out}"
ORACLE_ORACLE_MIN="${ORACLE_ORACLE_MIN:-0.01}"
ORACLE_RANDOM_MAX="${ORACLE_RANDOM_MAX:-0.35}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_DIR/pairwise"

read -r -a ORACLES <<< "${ORACLE_MASKS:-}"
if [[ ${#ORACLES[@]} -lt 1 ]]; then
  echo "Set ORACLE_MASKS to one or more .pt mask paths." >&2
  exit 2
fi

failures=0
declare -a oracle_random_pairs=()

for ((i = 0; i < ${#ORACLES[@]}; i++)); do
  for ((j = i + 1; j < ${#ORACLES[@]}; j++)); do
    a="${ORACLES[$i]}"
    b="${ORACLES[$j]}"
    tag="oracle_${i}_vs_${j}"
    echo "=== Oracle pair $tag ==="
    "$PYTHON_BIN" src/cold_start/mask_to_jaccard.py "$a" "$b" \
      --extended-aggregates both \
      -o "${OUT_DIR}/pairwise/jaccard_${tag}.json"
    agg="$("$PYTHON_BIN" -c "import json; d=json.load(open('${OUT_DIR}/pairwise/jaccard_${tag}.json')); print(d['jaccard']['aggregate'])")"
    awk -v j="$agg" -v m="$ORACLE_ORACLE_MIN" 'BEGIN { if (j < m) exit 1; exit 0 }' || {
      echo "FAIL: oracle-oracle aggregate Jaccard $agg < $ORACLE_ORACLE_MIN ($a vs $b)" >&2
      failures=$((failures + 1))
    }
  done
done

if [[ -n "${RANDOM_BASELINE_DIR:-}" ]]; then
  mkdir -p "$RANDOM_BASELINE_DIR"
  for ((i = 0; i < ${#ORACLES[@]}; i++)); do
    o="${ORACLES[$i]}"
    rnd="${RANDOM_BASELINE_DIR}/random_vs_oracle_${i}.pt"
    if [[ ! -f "$rnd" ]] && [[ -n "${REFERENCE_MASK_FOR_RANDOM:-}" ]]; then
      echo "Generating random baseline: $rnd"
      "$PYTHON_BIN" src/warm_start/random_mask_baseline.py \
        --reference_mask "$REFERENCE_MASK_FOR_RANDOM" \
        ${SPARSITY_FOR_RANDOM:+--sparsity_percent "$SPARSITY_FOR_RANDOM"} \
        --output_file "$rnd" \
        --seed $((42 + i))
    fi
    if [[ -f "$rnd" ]]; then
      tag="oracle_${i}_vs_random"
      echo "=== Oracle vs random $tag ==="
      "$PYTHON_BIN" src/cold_start/mask_to_jaccard.py "$o" "$rnd" \
        --extended-aggregates param_bucket \
        -o "${OUT_DIR}/pairwise/jaccard_${tag}.json"
      agg="$("$PYTHON_BIN" -c "import json; d=json.load(open('${OUT_DIR}/pairwise/jaccard_${tag}.json')); print(d['jaccard']['aggregate'])")"
      awk -v j="$agg" -v M="$ORACLE_RANDOM_MAX" 'BEGIN { if (j > M) exit 1; exit 0 }' || {
        echo "FAIL: oracle-random aggregate Jaccard $agg > $ORACLE_RANDOM_MAX ($o vs $rnd)" >&2
        failures=$((failures + 1))
      }
    else
      echo "Skip random test for oracle $i: set REFERENCE_MASK_FOR_RANDOM and SPARSITY_FOR_RANDOM to generate $rnd" >&2
    fi
  done
fi

echo "Done. Artifacts under $OUT_DIR/pairwise. Failure count: $failures"
exit "$failures"

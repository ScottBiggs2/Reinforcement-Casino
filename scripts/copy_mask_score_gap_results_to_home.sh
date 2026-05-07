#!/usr/bin/env bash
# Copy mask-score-gap outputs suitable for $HOME / laptop (excludes magnitude_caches).
# Only copies files under MAX_BYTES (default 1G) within parallel_shards/ so huge debug shards are skipped.
#
# Usage (Explorer login node):
#   bash scripts/copy_mask_score_gap_results_to_home.sh /scratch/$USER/rl_casino_analysis/mask_score_gap_parallel/run2_parallel
#   RESULTS_ROOT=$HOME/rl_casino/results bash scripts/copy_mask_score_gap_results_to_home.sh "$OUT_DIR" run2_parallel
#
# Then regenerate plots if needed:
#   PYTHONPATH=. python scripts/report_mask_score_gap_plots.py --analysis-dir "$HOME/rl_casino/results/run2_parallel"
#
set -euo pipefail

SRC="${1:?Usage: $0 OUT_DIR_ON_SCRATCH [DEST_NAME]}"
NAME="${2:-$(basename "$SRC")}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/rl_casino/results}"
DEST="${RESULTS_ROOT}/${NAME}"
MAX_BYTES=$((1024 * 1024 * 1024)) # 1 GiB

if [ ! -d "$SRC" ]; then
  echo "ERROR: not a directory: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST"

echo "SRC=$SRC"
echo "DEST=$DEST (no magnitude_caches/; parallel_shards files must be < ${MAX_BYTES} bytes)"

shopt -s nullglob
for f in "${SRC}"/*.csv "${SRC}"/*.json "${SRC}"/*.npz; do
  [ -f "$f" ] && cp -a "$f" "$DEST/"
done
shopt -u nullglob

if [ -d "${SRC}/figures" ]; then
  cp -a "${SRC}/figures" "$DEST/"
fi

if [ -d "${SRC}/parallel_shards" ]; then
  mkdir -p "${DEST}/parallel_shards"
  while IFS= read -r -d '' f; do
    cp -a "$f" "${DEST}/parallel_shards/"
  done < <(find "${SRC}/parallel_shards" -maxdepth 1 -type f -size -"${MAX_BYTES}"c -print0 2>/dev/null)
  _n=$(find "${DEST}/parallel_shards" -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "parallel_shards/: copied ${_n} files (each < 1G)"
fi

# Optional small text / json at top level not matched by glob
for f in mask_score_gap_run.json; do
  if [ -f "${SRC}/${f}" ] && [ ! -f "${DEST}/${f}" ]; then
    cp -a "${SRC}/${f}" "${DEST}/"
  fi
done

echo ""
du -sh "$DEST"
echo "Done. If figures/ is missing, cluster matplotlib often needs Agg backend — after git pull run:"
echo "  MPLBACKEND=Agg ANALYSIS_DIR=${DEST} bash scripts/run_mask_score_gap_plots.sh"

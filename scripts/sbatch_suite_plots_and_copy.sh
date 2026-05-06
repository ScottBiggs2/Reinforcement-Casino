#!/usr/bin/env bash
# Plot suite artifacts (CSV/probe JSON) and rsync into home results.
#
# Required env:
#   SUITE_DIR=/scratch/$USER/rl_casino_masks/<suite_out_dir>
#   DEST_DIR=/home/$USER/rl_casino/results/<name>
#
# Optional env:
#   PLOT_PY=/scratch/$USER/conda_envs/rl_casino/bin/python   (defaults to TRAIN_PY)
#
# This is intentionally non-interactive and quote-safe (no --wrap escaping).

#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=suite_plots_copy
#SBATCH --output=logs/suite_plots_copy_%j.out
#SBATCH --error=logs/suite_plots_copy_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

SUITE_DIR="${SUITE_DIR:?set SUITE_DIR=/scratch/$USER/...}"
DEST_DIR="${DEST_DIR:?set DEST_DIR=/home/$USER/rl_casino/results/...}"

PLOT_PY="${PLOT_PY:-$TRAIN_PY}"

test -f "${SUITE_DIR}/suite_summary.json"
mkdir -p "${SUITE_DIR}/plots" "${SUITE_DIR}/probe_plots"

echo "SUITE_DIR=$SUITE_DIR"
echo "DEST_DIR=$DEST_DIR"
echo "PLOT_PY=$PLOT_PY"

echo "=== plot_layer_metrics_csv ==="
"${PLOT_PY}" src/cold_start/plot_layer_metrics_csv.py \
  --input-dir "${SUITE_DIR}/pairwise" \
  --pattern "layer_metrics_*.csv" \
  --output-dir "${SUITE_DIR}/plots" \
  --jaccard-mc-trials 0 \
  --y-scale linear

echo "=== plot_probe_reports ==="
"${PLOT_PY}" src/cold_start/plot_probe_reports.py \
  --suite-dir "${SUITE_DIR}" \
  --output-dir "${SUITE_DIR}/probe_plots"

if [ -f "${SUITE_DIR}/dense_vs_mask_probes.json" ]; then
  echo "=== plot_dense_vs_mask_probes ==="
  "${PLOT_PY}" src/cold_start/plot_dense_vs_mask_probes.py \
    --input-json "${SUITE_DIR}/dense_vs_mask_probes.json" \
    --output-dir "${SUITE_DIR}/probe_plots"
fi

echo "=== rsync to results ==="
mkdir -p "$(dirname "${DEST_DIR}")"
rsync -a --delete "${SUITE_DIR}/" "${DEST_DIR}/"

echo "DONE. Results at: ${DEST_DIR}"
ls -lah "${DEST_DIR}/plots" "${DEST_DIR}/probe_plots" || true


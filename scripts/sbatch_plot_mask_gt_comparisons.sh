#!/bin/bash
# Regenerate mask-vs-GT layer_metrics PNGs on a compute node (do NOT run plot_layer_metrics_csv on the login node).
#
# Usage (from repo root, after layer_metrics_*.csv exist under comparisons_vs_ground_truth):
#   export MASK_ANALYSIS_DIR="/scratch/$USER/rl_casino_masks/..."
#   sbatch scripts/sbatch_plot_mask_gt_comparisons.sh
#
# Optional:
#   export PLOT_MASK_GT_REMOVE_OLD=1   # delete existing *_plots.png under plots/ before regenerating
#   export PLOT_Y_SCALE=log            # default in mask_gt suite
#   export CKA_TOTAL_N=...             # passed through if set
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=mask_gt_plots
#SBATCH --output=logs/mask_gt_plots_%j.out
#SBATCH --error=logs/mask_gt_plots_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

MASK_ANALYSIS_DIR="${MASK_ANALYSIS_DIR:-/scratch/${USER}/rl_casino_masks/tulu3_500_h200_fresh_0409_again}"
COMP_DIR="${MASK_ANALYSIS_DIR}/comparisons_vs_ground_truth"
PLOT_DIR="${COMP_DIR}/plots"

mkdir -p logs "$PLOT_DIR"

if [ ! -d "$COMP_DIR" ]; then
  echo "ERROR: COMP_DIR does not exist: $COMP_DIR" >&2
  exit 1
fi

if [ "${PLOT_MASK_GT_REMOVE_OLD:-0}" = "1" ]; then
  echo "Removing old *_plots.png under $PLOT_DIR"
  find "$PLOT_DIR" -maxdepth 1 -name '*_plots.png' -delete 2>/dev/null || true
fi

echo "COMP_DIR=$COMP_DIR"
echo "PLOT_DIR=$PLOT_DIR"
echo "Matched CSV count: $(find "$COMP_DIR" -maxdepth 1 -name 'layer_metrics_*.csv' ! -name '*_summary.csv' | wc -l)"

exec "$TRAIN_PY" src/cold_start/plot_layer_metrics_csv.py \
  --input-dir "$COMP_DIR" \
  --recursive \
  --pattern "layer_metrics_*.csv" \
  --output-dir "$PLOT_DIR" \
  --random-trials 1 \
  --random-seed 42 \
  --y-scale "${PLOT_Y_SCALE:-log}" \
  ${CKA_TOTAL_N:+--cka-total-n "${CKA_TOTAL_N}"} \
  --log-y-floor "${PLOT_LOG_Y_FLOOR:-1e-12}"

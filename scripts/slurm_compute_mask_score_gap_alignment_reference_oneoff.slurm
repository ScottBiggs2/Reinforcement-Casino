#!/usr/bin/env bash
# One-off CPU job: compute alignment-optimal s* reference sidecars for an existing mask-score-gap run.
#
# Requires env:
#   OUT_DIR   existing analysis dir (contains mask_score_gap_run.json + magnitude_caches/)
#   FISHER    path to fisher diag .pt dict (name->tensor) (e.g. produced by slurm_export_alignment_fisher_diag_oneoff)
#   ALIGN_C           constant C in s*(theta,B)
#   ALIGN_EPS_GRAD    epsilon (gradient bound) in s*(theta,B)
#
# Optional:
#   ALIGN_ETA=1e-6, ALIGN_MAX_ITER=25, HISTOGRAM_BINS=1024
#   ALIGN_MAX_KEYS=... (smoke)
#
# Example (login node):
#   export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_parallel/run2_parallel
#   export FISHER=/scratch/$USER/rl_casino_analysis/alignment/fisher_diag_raw_ckpt500_a100.pt
#   export ALIGN_C=1.0 ALIGN_EPS_GRAD=0.0
#   sbatch scripts/slurm_compute_mask_score_gap_alignment_reference_oneoff.slurm
#
# Output (written into OUT_DIR):
#   mask_score_gap_alignment_gap_diagnostics.json
#   mask_score_gap_alignment_histograms.npz

#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --job-name=mgap_align
#SBATCH --output=logs/mask_score_gap_alignment_%j.out
#SBATCH --error=logs/mask_score_gap_alignment_%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

OUT_DIR="${OUT_DIR:?set OUT_DIR}"
FISHER="${FISHER:?set FISHER}"
ALIGN_C="${ALIGN_C:?set ALIGN_C}"
ALIGN_EPS_GRAD="${ALIGN_EPS_GRAD:?set ALIGN_EPS_GRAD}"

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY} (set TRAIN_ENV)" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

HISTOGRAM_BINS="${HISTOGRAM_BINS:-1024}"
ALIGN_ETA="${ALIGN_ETA:-1e-6}"
ALIGN_MAX_ITER="${ALIGN_MAX_ITER:-25}"
ALIGN_MAX_KEYS="${ALIGN_MAX_KEYS:-}"

ARGS=(scripts/compute_mask_score_gap_alignment_reference.py
  --analysis-dir "${OUT_DIR}"
  --fisher-diag "${FISHER}"
  --C "${ALIGN_C}"
  --epsilon-grad "${ALIGN_EPS_GRAD}"
  --eta "${ALIGN_ETA}"
  --max-iter "${ALIGN_MAX_ITER}"
  --histogram-bins "${HISTOGRAM_BINS}"
)
if [ -n "${ALIGN_MAX_KEYS}" ]; then
  ARGS+=(--max-keys "${ALIGN_MAX_KEYS}")
fi

echo "Running: ${TRAIN_PY} ${ARGS[*]}"
"${TRAIN_PY}" "${ARGS[@]}"

echo "Done. Sidecars under OUT_DIR=${OUT_DIR}"


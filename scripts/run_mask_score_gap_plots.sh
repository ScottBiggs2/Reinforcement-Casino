#!/usr/bin/env bash
# Run matplotlib report for an analysis directory (local machine or HPC login node).
#
# Usage:
#   ANALYSIS_DIR=$HOME/rl-casino/results/run2_parallel bash scripts/run_mask_score_gap_plots.sh
#
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
cd "$REPO_ROOT"

ANALYSIS_DIR="${ANALYSIS_DIR:?set ANALYSIS_DIR=...}"
OUT_DIR_PLOTS="${OUT_DIR_PLOTS:-}"

_SCRATCH="${SCRATCH_USER_ROOT:-/scratch/${USER:-}}"
_DEFAULT_PY="${_SCRATCH}/conda_envs/rl_casino/bin/python"
if [ -x "${TRAIN_PY:-}" ]; then
  PY="${TRAIN_PY}"
elif [ -x "$_DEFAULT_PY" ]; then
  PY="$_DEFAULT_PY"
else
  PY="python3"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ARGS=(scripts/report_mask_score_gap_plots.py --analysis-dir "${ANALYSIS_DIR}")
if [ -n "${OUT_DIR_PLOTS}" ]; then
  ARGS+=(--out-dir "${OUT_DIR_PLOTS}")
fi

exec "$PY" "${ARGS[@]}"

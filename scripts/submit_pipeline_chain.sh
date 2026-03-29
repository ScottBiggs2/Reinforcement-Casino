#!/usr/bin/env bash
# Launch the full pipeline as a chain of Slurm jobs (each ≤8h by default) with afterok dependencies.
# Run from the repo root on a login node:
#   bash scripts/submit_pipeline_chain.sh
#
# Requires: sbatch from login/compute; cluster allows job dependencies and nested sbatch.
#
# Exports PIPELINE_RUN_ID and RUN_ID for all stages (same run id for paths under scratch).
# Override hyperparameters via environment variables before calling this script (same as pipeline_common.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-$(date +%Y%m%d_%H%M%S)_manual}"
export RUN_ID="${PIPELINE_RUN_ID}"

echo "PIPELINE_RUN_ID=${PIPELINE_RUN_ID}"
echo "Submitting stage 1 (dense DPO)…"

JID=$(sbatch --parsable \
  --export=ALL,PIPELINE_RUN_ID,RUN_ID \
  "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh")

echo "Stage 1 job id: ${JID}"
echo "Watch:  squeue -u \"\$USER\""
echo "Stage1 log (after job starts):  tail -f logs/pipeline_${JID}_p1_dense.out"
echo "All stages use RUN_ID=${PIPELINE_RUN_ID} under your scratch roots (see pipeline_common.sh)."

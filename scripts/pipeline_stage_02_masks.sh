#!/bin/bash
# Stage 2 entry (CPU, ~30m allocation): submits the split mask pipeline 02a→02b→02c→3.
#
# This script does NOT run mask Python and does NOT request a GPU. Mixed CPU/GPU mask work
# is split across dedicated jobs (see pipeline_stage_02a_masks_warm.sh, …02b…, …02c…).
# Kept as pipeline_stage_02_masks.sh for resume compatibility (e.g. resume stage 2all).
#
# Northeastern Explorer CPU batch partition; override: sbatch -p … script.sh
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=pipe_p2_entry
#SBATCH --output=logs/pipeline_%j_p2_masks_entry.out
#SBATCH --error=logs/pipeline_%j_p2_masks_entry.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

if [ -z "${RUN_ID:-}" ] && [ -n "${PIPELINE_RUN_ID:-}" ]; then
  export RUN_ID="${PIPELINE_RUN_ID}"
fi
if [ -z "${RUN_ID:-}" ]; then
  echo "ERROR: RUN_ID or PIPELINE_RUN_ID must be set (pass via sbatch --export)" >&2
  exit 1
fi
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-$RUN_ID}"

echo "===== Stage 2 entry: submitting split mask chain (02a→02b→02c→3) RUN_ID=${RUN_ID} ====="
jid=$(sbatch --parsable \
  --partition="${CPU_PARTITION:-short}" \
  --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
  "${REPO_ROOT}/scripts/pipeline_stage_02a_masks_warm.sh")
echo "Submitted pipeline_stage_02a_masks_warm.sh → job ${jid}"
echo "Follow logs: logs/pipeline_${jid}_p2a_masks_warm.out (then p2b / p2c as each stage chains)."

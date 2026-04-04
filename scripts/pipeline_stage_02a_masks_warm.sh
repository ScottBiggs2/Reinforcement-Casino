#!/bin/bash
# Stage 2a/5: warm masks only (delta-based; CPU partition — no idle-GPU risk). Chains to 2b (cold GPU).
# Northeastern Explorer: `short` (not `cpu`). Override: sbatch -p … this script.
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:45:00
#SBATCH --job-name=pipe_p2a_warm
#SBATCH --mem=128G
#SBATCH --output=logs/pipeline_%j_p2a_masks_warm.out
#SBATCH --error=logs/pipeline_%j_p2a_masks_warm.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

export PIPELINE_MASK_PHASE=warm
# No GPU on this partition — keep scoring on CPU (override only if your site maps CPU job to GPU node).
export RL_CASINO_WARM_MASK_SCORE_DEVICE="${RL_CASINO_WARM_MASK_SCORE_DEVICE:-cpu}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

echo "===== STAGE 2a/5: warm masks only (${RUN_ID}) ====="
run_masks

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: expected SLURM_JOB_ID for chaining" >&2
  exit 1
fi
jid=$(sbatch --parsable \
  --dependency=afterok:"${SLURM_JOB_ID}" \
  --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
  "${REPO_ROOT}/scripts/pipeline_stage_02b_masks_cold.sh")
echo "Chained next stage: pipeline_stage_02b_masks_cold.sh → Slurm job ${jid} (afterok:${SLURM_JOB_ID})"

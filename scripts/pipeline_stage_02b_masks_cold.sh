#!/bin/bash
# Stage 2b/5: cold Fisher + CAV (GPU — full model forward). Chains to 2c (random + inverses on CPU).
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --job-name=pipe_p2b_cold
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p2b_masks_cold.out
#SBATCH --error=logs/pipeline_%j_p2b_masks_cold.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

export PIPELINE_MASK_PHASE=cold

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

echo "===== STAGE 2b/5: cold Fisher + CAV (${RUN_ID}) ====="
run_masks

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: expected SLURM_JOB_ID for chaining" >&2
  exit 1
fi
jid=$(sbatch --parsable \
  --dependency=afterok:"${SLURM_JOB_ID}" \
  --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
  "${REPO_ROOT}/scripts/pipeline_stage_02c_masks_post.sh")
echo "Chained next stage: pipeline_stage_02c_masks_post.sh → Slurm job ${jid} (afterok:${SLURM_JOB_ID})"

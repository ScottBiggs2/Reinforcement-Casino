#!/bin/bash
# Stage 5/5: submit benchmark eval jobs (baseline / dense / sparse). Short wall — only sbatch fan-out.
# CPU-only: eval workers are separate GPU jobs from run_evals_slurm.sh.
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=pipe_p5_eval
#SBATCH --mem=16G
#SBATCH --output=logs/pipeline_%j_p5_eval.out
#SBATCH --error=logs/pipeline_%j_p5_eval.err

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

echo "===== STAGE 5/5: eval submissions (${RUN_ID}) ====="
submit_evals
echo "===== PIPELINE CHAIN COMPLETE (${RUN_ID}) ====="

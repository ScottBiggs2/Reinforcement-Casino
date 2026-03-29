#!/bin/bash
# Stage 4: submit one Slurm job per mask .pt (parallel sparse DPO, 2k steps / tulu3 via pipeline_common).
# Then submits stage 5 (evals) with dependency on all sparse jobs (afterok by default).
# This launcher exits quickly once submissions are queued; check logs/sparse_<RUN_ID>_*.out per mask.
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --job-name=pipe_p4_launch
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p4_sparse_launch.out
#SBATCH --error=logs/pipeline_%j_p4_sparse_launch.err

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

echo "===== STAGE 4: parallel sparse DPO launch (${RUN_ID}) ====="
echo "Per-mask wall time: ${SPARSE_SLURM_TIME:-08:00:00}  |  eval dependency: ${PIPELINE_SPARSE_EVAL_DEPENDENCY:-afterok}"
launch_parallel_sparse_jobs_and_eval
echo "===== Sparse launch + eval submit complete (${RUN_ID}) ====="

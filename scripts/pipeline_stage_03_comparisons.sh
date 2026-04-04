#!/bin/bash
# Stage 3/5 (legacy GPU): mask comparisons (Jaccard / optional CKA / CSV / plots).
# Prefer pipeline_stage_03_comparisons_cpu.sh + optional 3b. Stage 4 is kicked off at the start of stage 3.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=07:45:00
#SBATCH --job-name=pipe_p3_cmp
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p3_cmp.out
#SBATCH --error=logs/pipeline_%j_p3_cmp.err

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

echo "===== STAGE 3/5: mask comparisons (${RUN_ID}) ====="
pipeline_submit_sparse_stage_early
run_mask_comparisons

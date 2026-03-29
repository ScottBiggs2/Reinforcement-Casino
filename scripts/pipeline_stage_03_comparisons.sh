#!/bin/bash
# Stage 3/5: mask comparisons (Jaccard / optional CKA / CSV / plots).
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=pipe_p3_cmp
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p3_cmp.out
#SBATCH --error=logs/pipeline_%j_p3_cmp.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/pipeline_common.sh"
pipeline_setup

echo "===== STAGE 3/5: mask comparisons (${RUN_ID}) ====="
run_mask_comparisons
pipeline_submit_next_stage "pipeline_stage_04_sparse.sh"

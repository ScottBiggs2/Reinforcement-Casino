#!/bin/bash
# Stage 1/5: dense DPO (+ delta logs). Chains to stage 2 on success.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=pipe_p1_dense
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p1_dense.out
#SBATCH --error=logs/pipeline_%j_p1_dense.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/pipeline_common.sh"
pipeline_setup

echo "===== STAGE 1/5: dense DPO (${RUN_ID}) ====="
run_dense_dpo
pipeline_submit_next_stage "pipeline_stage_02_masks.sh"

#!/bin/bash
# Stage 2/5: warm + cold masks.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=pipe_p2_masks
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p2_masks.out
#SBATCH --error=logs/pipeline_%j_p2_masks.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/pipeline_common.sh"
pipeline_setup

echo "===== STAGE 2/5: masks (${RUN_ID}) ====="
run_masks
pipeline_submit_next_stage "pipeline_stage_03_comparisons.sh"

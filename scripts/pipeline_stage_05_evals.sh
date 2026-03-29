#!/bin/bash
# Stage 5/5: submit benchmark eval jobs (baseline / dense / sparse). Short wall — only sbatch fan-out.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=pipe_p5_eval
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p5_eval.out
#SBATCH --error=logs/pipeline_%j_p5_eval.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/pipeline_common.sh"
pipeline_setup

echo "===== STAGE 5/5: eval submissions (${RUN_ID}) ====="
submit_evals
echo "===== PIPELINE CHAIN COMPLETE (${RUN_ID}) ====="

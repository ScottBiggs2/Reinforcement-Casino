#!/bin/bash
# Single Slurm job: run the entire pipeline in one allocation (needs a long enough wall time).
# For clusters with an ~8h cap, use: bash scripts/submit_pipeline_chain.sh
#
# Usage (repo root): sbatch scripts/run_full_pipeline.sh
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --job-name=full_pipeline
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err

set -euo pipefail

# Slurm copies this script to /var/spool/slurmd/.../slurm_script — use submit directory as repo root.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# Monolithic defaults (override 8h-oriented defaults in pipeline_common.sh unless env already set)
export TRAIN_TIMEOUT_PER_DATASET="${TRAIN_TIMEOUT_PER_DATASET:-$((16 * 60 * 60))}"
export MASK_TIMEOUT="${MASK_TIMEOUT:-$((4 * 60 * 60))}"
export SPARSE_TIMEOUT_PER_MASK="${SPARSE_TIMEOUT_PER_MASK:-$((4 * 60 * 60))}"
export GLOBAL_MAX_SECONDS="${GLOBAL_MAX_SECONDS:-$((47 * 60 * 60))}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

echo "===== FULL PIPELINE START (${RUN_ID}) ====="
run_dense_dpo
run_masks
run_mask_comparisons
run_sparse_dpo
submit_evals
echo "===== FULL PIPELINE COMPLETE (${RUN_ID}) ====="

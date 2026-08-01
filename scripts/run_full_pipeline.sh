#!/bin/bash
# Single Slurm job: dense DPO → masks → (sparse jobs submitted early) → comparisons → done.
# Sparse training runs in separate GPU jobs; eval fan-out is stage 5 (queued by the sparse launcher).
# Typical clusters cap jobs at 8h — full pipelines usually need the chained pipeline instead.
# Prefer: bash scripts/submit_pipeline_chain.sh
#
# Usage (repo root): sbatch scripts/run_full_pipeline.sh
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
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

# Per-stage budgets come from pipeline_common.sh (~7h30 per stage, ~7h45 global) so one allocation
# stays under a typical 8h Slurm cap. For longer dense/mask/sparse runs, use scripts/submit_pipeline_chain.sh.

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

echo "===== FULL PIPELINE START (${RUN_ID}) ====="
run_dense_dpo
run_masks
pipeline_submit_sparse_stage_early
run_mask_comparisons
echo "===== FULL PIPELINE COMPLETE (${RUN_ID}) (sparse + evals run via nested Slurm from stage 4 launcher) ====="

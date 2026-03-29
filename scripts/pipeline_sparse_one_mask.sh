#!/bin/bash
# One sparse DPO run (NUM_STEPS_DPO, tulu3 / DPO_DATASETS[0]) for a single mask .pt.
# Submitted in parallel by pipeline_stage_04_sparse.sh. Logs: logs/sparse_<RUN_ID>_<mask>_*.out
#
# Required env: PIPELINE_MASK_FILE (absolute path to .pt), RUN_ID / PIPELINE_RUN_ID
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --job-name=sparse_one
#SBATCH --mem=128G
#SBATCH --ntasks=1

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

# Per-job training timeout should match wall time (pipeline_common defaults are for 8h chain stages).
export SPARSE_TIMEOUT_PER_MASK="${SPARSE_TIMEOUT_PER_MASK:-$((23 * 60 * 60))}"
export GLOBAL_MAX_SECONDS="${GLOBAL_MAX_SECONDS:-$((23 * 60 * 60 + 30 * 60))}"

pipeline_setup

if [ -z "${PIPELINE_MASK_FILE:-}" ] || [ ! -f "${PIPELINE_MASK_FILE}" ]; then
  echo "ERROR: PIPELINE_MASK_FILE must name an existing .pt mask (got: ${PIPELINE_MASK_FILE:-})" >&2
  exit 1
fi

echo "===== Sparse DPO single mask: ${PIPELINE_MASK_FILE} (RUN_ID=${RUN_ID}) ====="
run_sparse_dpo_one_mask "${PIPELINE_MASK_FILE}"
echo "===== Sparse DPO finished for $(basename "${PIPELINE_MASK_FILE}") ====="

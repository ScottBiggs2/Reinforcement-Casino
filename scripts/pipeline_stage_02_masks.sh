#!/bin/bash
# Stage 2/5: warm + cold masks.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=pipe_p2_masks
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p2_masks.out
#SBATCH --error=logs/pipeline_%j_p2_masks.err

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

echo "===== STAGE 2/5: masks (${RUN_ID}) ====="
run_masks
# Stage 3a: CPU by default (Jaccard/CSV/plots). If RUN_MASK_CKA=1, stage 3a will chain 3b (GPU CKA).
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: expected SLURM_JOB_ID for chaining" >&2
  exit 1
fi
jid=$(sbatch --parsable \
  --dependency=afterok:"${SLURM_JOB_ID}" \
  --partition="${CPU_PARTITION:-cpu}" \
  --time="${PIPELINE_CPU_COMPARISON_TIME:-06:00:00}" \
  --mem="${PIPELINE_CPU_COMPARISON_MEM:-64G}" \
  --cpus-per-task="${PIPELINE_CPU_COMPARISON_CPUS:-4}" \
  --ntasks=1 \
  --output=logs/pipeline_%j_p3_cmp_cpu.out \
  --error=logs/pipeline_%j_p3_cmp_cpu.err \
  --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
  "${REPO_ROOT}/scripts/pipeline_stage_03_comparisons_cpu.sh")
echo "Chained next stage: pipeline_stage_03_comparisons_cpu.sh → Slurm job ${jid} (afterok:${SLURM_JOB_ID})"

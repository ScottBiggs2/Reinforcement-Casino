#!/bin/bash
# Stage 3/5 (CPU): mask comparisons (Jaccard / CSV / plots).
# This stage does NOT require a GPU unless RUN_MASK_CKA=1; on some clusters, GPU jobs that
# don't use the GPU get cancelled. Use this CPU flavor when RUN_MASK_CKA=0.
#
# This script intentionally omits #SBATCH headers so it can be submitted with `sbatch -p ...`
# from another stage (see pipeline_stage_02_masks.sh / resume_pipeline_from_stage.sh).

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

echo "===== STAGE 3a/5 (CPU): mask comparisons (${RUN_ID}) ====="

# CPU-safe portion: Jaccard / CSV / plots. (CKA is delegated to stage 3b on GPU.)
run_mask_comparisons

if [ "${RUN_MASK_CKA:-0}" = "1" ]; then
  echo "CKA enabled (RUN_MASK_CKA=1): chaining stage 3b (GPU CKA)."
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: expected SLURM_JOB_ID for chaining" >&2
    exit 1
  fi
  jid=$(sbatch --parsable \
    --dependency=afterok:"${SLURM_JOB_ID}" \
    --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
    "${REPO_ROOT}/scripts/pipeline_stage_03b_cka_gpu.sh")
  echo "Chained next stage: pipeline_stage_03b_cka_gpu.sh → Slurm job ${jid} (afterok:${SLURM_JOB_ID})"
else
  pipeline_submit_next_stage "pipeline_stage_04_sparse.sh"
fi


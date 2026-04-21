#!/usr/bin/env bash
# Slurm batch body for jobs queued by orchestrate_masks_then_queue_dpo_grpo.slurm.
# No #SBATCH lines here — resources are passed on the sbatch command line when using
# ORCH_USE_TRAIN_AUTO_RESUME=1 so embedded directives in the inner script are not lost.
#
# Env:
#   ORCH_TRAIN_INNER   absolute path to pipeline_stage_01_dense.sh, pipeline_sparse_one_mask.sh,
#                      or grpo_openr1_llama31_slurm.sh
#   ORCH_USE_TRAIN_AUTO_RESUME  if 1, exec train_with_auto_resume.sh with that inner; else exec inner only
#   AUTO_RESUME_MODE   dense_dpo | sparse_dpo | dense_grpo | sparse_grpo (required when using wrapper)

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"

INNER="${ORCH_TRAIN_INNER:?ORCH_TRAIN_INNER must be set to the absolute path of the inner training script}"

if [ "${ORCH_USE_TRAIN_AUTO_RESUME:-0}" = "1" ]; then
  exec bash "${REPO_ROOT}/scripts/train_with_auto_resume.sh" "${INNER}"
else
  exec bash "${INNER}"
fi

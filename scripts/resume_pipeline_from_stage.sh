#!/usr/bin/env bash
# Resume a chained pipeline (submit_pipeline_chain.sh) after a stage failed or hit the wall clock.
# Stages 1–2 stay on disk under your scratch roots; you re-submit from the stage you need.
#
# Usage (from repo root on the login node):
#   bash scripts/resume_pipeline_from_stage.sh <3|4|5> <PIPELINE_RUN_ID>
#
# Examples:
#   bash scripts/resume_pipeline_from_stage.sh 3 20260329_185704_manual
#   # Skip comparisons if masks exist and you only want sparse + evals:
#   bash scripts/resume_pipeline_from_stage.sh 4 20260329_185704_manual
#   # Only eval fan-out (dense/sparse checkpoints already present):
#   bash scripts/resume_pipeline_from_stage.sh 5 20260329_185704_manual
#
# Stage 3 is often wall-limited when RUN_MASK_CKA=1. For a faster retry:
#   RUN_MASK_CKA=0 bash scripts/resume_pipeline_from_stage.sh 3 YOUR_RUN_ID
#
# Optional env (same as pipeline_common.sh): RUN_MASK_CKA, EVAL_LIMIT, etc.

set -euo pipefail

STAGE="${1:?usage: $0 <3|4|5> <PIPELINE_RUN_ID>}"
RUN="${2:?usage: $0 <3|4|5> <PIPELINE_RUN_ID>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

case "$STAGE" in
  3) NEXT="pipeline_stage_03_comparisons.sh" ;;
  4) NEXT="pipeline_stage_04_sparse.sh" ;;
  5) NEXT="pipeline_stage_05_evals.sh" ;;
  *)
    echo "ERROR: stage must be 3, 4, or 5 (got: ${STAGE})" >&2
    exit 1
    ;;
esac

if [ ! -f "${REPO_ROOT}/scripts/${NEXT}" ]; then
  echo "ERROR: missing ${REPO_ROOT}/scripts/${NEXT}" >&2
  exit 1
fi

echo "Resuming pipeline at stage ${STAGE}: ${NEXT}"
echo "  PIPELINE_RUN_ID=${RUN}"
echo "  (paths use RUN_ID from this id — same as your original chain)"

JID=$(sbatch --parsable \
  --export=ALL,PIPELINE_RUN_ID="${RUN}",RUN_ID="${RUN}" \
  "${REPO_ROOT}/scripts/${NEXT}")

echo "Submitted job ${JID}"
case "$STAGE" in
  3) _log="pipeline_${JID}_p3_cmp.out" ;;
  4) _log="pipeline_${JID}_p4_sparse_launch.out" ;;
  5) _log="pipeline_${JID}_p5_eval.out" ;;
esac
echo "  tail -f logs/${_log}"
echo "  squeue -u \"\$USER\""

#!/usr/bin/env bash
# Resume a chained pipeline (submit_pipeline_chain.sh) after a stage failed or hit the wall clock.
# Stages 1–2 stay on disk under your scratch roots; you re-submit from the stage you need.
#
# Usage (from repo root on the login node):
#   bash scripts/resume_pipeline_from_stage.sh <stage> <PIPELINE_RUN_ID>
#
# Stage codes:
#   2 | 2a   — warm masks only (GPU), then chain 2b→2c→3…
#   2b       — cold Fisher + CAV only (GPU), then chain 2c→3…
#   2c       — random + inverse masks only (CPU), then chain 3…
#   2all     — stage 2 entry script (submits 02a→02b→02c)
#   3 | 4 | 5 — comparisons, sparse launcher, evals
#
# Examples:
#   bash scripts/resume_pipeline_from_stage.sh 2a 20260329_185704_manual
#   bash scripts/resume_pipeline_from_stage.sh 2b 20260329_185704_manual
#   bash scripts/resume_pipeline_from_stage.sh 3 20260329_185704_manual
#   bash scripts/resume_pipeline_from_stage.sh 4 20260329_185704_manual
#   bash scripts/resume_pipeline_from_stage.sh 5 20260329_185704_manual
#
# Stage 3 is often wall-limited when RUN_MASK_CKA=1. For a faster retry:
#   RUN_MASK_CKA=0 bash scripts/resume_pipeline_from_stage.sh 3 YOUR_RUN_ID
#
# Stage 3 kicks off stage 4 (sparse) at job start — do NOT also run `resume … 4` unless you know stage 4 failed
# and you removed MASK_OUT_BASE/<id>/.sparse_launch_submitted (duplicate launches waste GPU).
# Re-run stage 3 without re-queuing sparse: PIPELINE_SKIP_SPARSE_LAUNCH=1 bash scripts/resume_pipeline_from_stage.sh 3 YOUR_RUN_ID
# Force a new stage 4 after cleaning outputs: PIPELINE_FORCE_SPARSE_RELUNCH=1 bash scripts/resume_pipeline_from_stage.sh 4 YOUR_RUN_ID
#
# Optional env (same as pipeline_common.sh): RUN_MASK_CKA, EVAL_LIMIT, PIPELINE_SKIP_SPARSE_LAUNCH,
# CPU_PARTITION (default short on Northeastern Explorer), GPU_PARTITION (default gpu), etc.

set -euo pipefail

STAGE="${1:?usage: $0 <2|2a|2b|2c|2all|3|4|5> <PIPELINE_RUN_ID>}"
RUN="${2:?usage: $0 <2|2a|2b|2c|2all|3|4|5> <PIPELINE_RUN_ID>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# Match pipeline_common.sh defaults for stage 3a (CPU comparisons)
PIPELINE_CPU_COMPARISON_TIME="${PIPELINE_CPU_COMPARISON_TIME:-07:45:00}"
PIPELINE_CPU_COMPARISON_MEM="${PIPELINE_CPU_COMPARISON_MEM:-128G}"
PIPELINE_CPU_COMPARISON_CPUS="${PIPELINE_CPU_COMPARISON_CPUS:-16}"
# Northeastern Explorer: CPU batch partition is `short` (not `cpu`). Override if your site differs.
CPU_PARTITION="${CPU_PARTITION:-short}"
GPU_PARTITION="${GPU_PARTITION:-gpu}"

NEXT=""

case "$STAGE" in
  2|2a) NEXT="pipeline_stage_02a_masks_warm.sh" ;;
  2b) NEXT="pipeline_stage_02b_masks_cold.sh" ;;
  2c) NEXT="pipeline_stage_02c_masks_post.sh" ;;
  2all) NEXT="pipeline_stage_02_masks.sh" ;;
  3) NEXT="pipeline_stage_03_comparisons.sh" ;;
  4) NEXT="pipeline_stage_04_sparse.sh" ;;
  5) NEXT="pipeline_stage_05_evals.sh" ;;
  *)
    echo "ERROR: stage must be 2, 2a, 2b, 2c, 2all, 3, 4, or 5 (got: ${STAGE})" >&2
    exit 1
    ;;
esac

if [ -n "${NEXT}" ] && [ ! -f "${REPO_ROOT}/scripts/${NEXT}" ]; then
  echo "ERROR: missing ${REPO_ROOT}/scripts/${NEXT}" >&2
  exit 1
fi

echo "Resuming pipeline at stage ${STAGE}: ${NEXT:-stage 3 comparisons}"
echo "  PIPELINE_RUN_ID=${RUN}"
echo "  (paths use RUN_ID from this id — same as your original chain)"

# Login-node guard: stage 4 is normally submitted by stage 3; refuse duplicate resume 4.
if [ "${STAGE}" = "4" ]; then
  SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
  MASK_OUT_BASE="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
  _sfile="${MASK_OUT_BASE}/${RUN}/.sparse_launch_submitted"
  if [ -f "${_sfile}" ] && [ "${PIPELINE_FORCE_SPARSE_RELUNCH:-0}" != "1" ]; then
    echo "ERROR: Stage 4 already completed for RUN_ID=${RUN} (${_sfile})." >&2
    echo "  Do not run resume 4 again — stage 3 already queues sparse at its start." >&2
    echo "  To redo: scancel duplicate jobs, remove sparse outputs, rm -f \"${_sfile}\", then:" >&2
    echo "    PIPELINE_FORCE_SPARSE_RELUNCH=1 bash scripts/resume_pipeline_from_stage.sh 4 ${RUN}" >&2
    exit 1
  fi
fi

if [ "${STAGE}" = "3" ] && [ "${RUN_MASK_CKA:-0}" != "1" ]; then
  JID=$(sbatch --parsable \
    --partition="${CPU_PARTITION}" \
    --time="${PIPELINE_CPU_COMPARISON_TIME}" \
    --mem="${PIPELINE_CPU_COMPARISON_MEM}" \
    --cpus-per-task="${PIPELINE_CPU_COMPARISON_CPUS}" \
    --ntasks=1 \
    --output=logs/pipeline_%j_p3_cmp_cpu.out \
    --error=logs/pipeline_%j_p3_cmp_cpu.err \
    --export=ALL,PIPELINE_RUN_ID="${RUN}",RUN_ID="${RUN}" \
    "${REPO_ROOT}/scripts/pipeline_stage_03_comparisons_cpu.sh")
elif [ "${STAGE}" = "3" ]; then
  JID=$(sbatch --parsable \
    --partition="${CPU_PARTITION}" \
    --time="${PIPELINE_CPU_COMPARISON_TIME}" \
    --mem="${PIPELINE_CPU_COMPARISON_MEM}" \
    --cpus-per-task="${PIPELINE_CPU_COMPARISON_CPUS}" \
    --ntasks=1 \
    --output=logs/pipeline_%j_p3_cmp_cpu.out \
    --error=logs/pipeline_%j_p3_cmp_cpu.err \
    --export=ALL,PIPELINE_RUN_ID="${RUN}",RUN_ID="${RUN}" \
    "${REPO_ROOT}/scripts/pipeline_stage_03_comparisons_cpu.sh")
else
  case "${STAGE}" in
    2|2a|2b) _sbatch_partition=(--partition="${GPU_PARTITION}") ;;
    *) _sbatch_partition=(--partition="${CPU_PARTITION}") ;;
  esac
  JID=$(sbatch --parsable \
    "${_sbatch_partition[@]}" \
    --export=ALL,PIPELINE_RUN_ID="${RUN}",RUN_ID="${RUN}" \
    "${REPO_ROOT}/scripts/${NEXT}")
fi

echo "Submitted job ${JID}"

case "$STAGE" in
  2|2a) _log="pipeline_${JID}_p2a_masks_warm.out" ;;
  2b) _log="pipeline_${JID}_p2b_masks_cold.out" ;;
  2c) _log="pipeline_${JID}_p2c_masks_post.out" ;;
  2all) _log="pipeline_${JID}_p2_masks_entry.out" ;;
  3) _log="pipeline_${JID}_p3_cmp_cpu.out" ;;
  4) _log="pipeline_${JID}_p4_sparse_launch.out" ;;
  5) _log="pipeline_${JID}_p5_eval.out" ;;
esac
echo "  tail -f logs/${_log}"
echo "  squeue -u \"\$USER\""

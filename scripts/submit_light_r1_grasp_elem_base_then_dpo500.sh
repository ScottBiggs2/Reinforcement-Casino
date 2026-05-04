#!/usr/bin/env bash
# Login node: submit Light-R1 GraSP element/base mask GPU job, then sparse DPO (500 steps) with afterok.
# Uses repo Slurm scripts so stdout/stderr land in logs/ (same as months-ago pipeline), not slurm-JOBID.out.
#
# Usage (repo root):
#   bash scripts/submit_light_r1_grasp_elem_base_then_dpo500.sh
#
# Optional overrides (export before running):
#   SCRATCH_USER_ROOT  HF_TOKEN  MODEL  MASK_SLURM_PARTITION  MASK_SLURM_GRES  SPARSE_SLURM_PARTITION  SPARSE_SLURM_GRES
# To reuse a fixed MASK_RUN_ID / RUN_ID from the shell: SUBMIT_LR1_GRASP_REUSE_IDS=1 bash ...
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"

# Stale login exports caused doubled RUN_ID and wrong MASK_RUN_ID; clear unless explicitly reusing.
if [ "${SUBMIT_LR1_GRASP_REUSE_IDS:-0}" != "1" ]; then
  unset RUN_ID PIPELINE_RUN_ID MASK_RUN_ID MASK_OUT_BASE 2>/dev/null || true
fi
# sbatch --export=ALL forwards login TRAIN_ENV; a stale broken value breaks mask + sparse jobs.
unset TRAIN_ENV TRAIN_PY 2>/dev/null || true

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/export_pipeline_step_targets.sh"

export MASK_RUN_ID="${MASK_RUN_ID:-lr1_gr_elem_base_$(date +%Y%m%d_%H%M%S)}"
export RUN_ID="${RUN_ID:-dpo500_sparse_lr1_grasp_elem_base_${MASK_RUN_ID}}"
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-${RUN_ID}}"
export DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-${DPO_DS_KEY_LIGHT_R1}}"

# Match working GRPO on Explorer: orchestrate_grpo_500step_5way.slurm + mask suite use multigpu + h200.
export MASK_SLURM_PARTITION="${MASK_SLURM_PARTITION:-multigpu}"
export MASK_SLURM_GRES="${MASK_SLURM_GRES:-gpu:h200:1}"
export SPARSE_SLURM_PARTITION="${SPARSE_SLURM_PARTITION:-multigpu}"
export SPARSE_SLURM_GRES="${SPARSE_SLURM_GRES:-gpu:h200:1}"

export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-100}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"

sanitize_model_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s//-/_}"
  s="${s,,}"
  s="$(echo "$s" | tr -c '[:alnum:]_' '_' )"
  while [[ "$s" == *"__"* ]]; do s="${s//__/_}"; done
  s="${s##_}"
  s="${s%%_}"
  echo "$s"
}

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SANITIZED="$(sanitize_model_name "$MODEL")"
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export GRASP_OBJECTIVE="${GRASP_OBJECTIVE:-dpo_preference}"

MASK_DIR="${SCRATCH_USER_ROOT}/rl_casino_masks/${MASK_RUN_ID}"
MASK_GRASP_ELEM_BASE="${MASK_DIR}/grasp_${MODEL_SANITIZED}_light_r1_sp${SPARSITY_PERCENT}pct_obj${GRASP_OBJECTIVE}_elem_base.pt"
mkdir -p "${MASK_DIR}"

echo "MASK_RUN_ID=${MASK_RUN_ID}"
echo "RUN_ID=${RUN_ID}"
echo "PIPELINE_MASK_FILE will be: ${MASK_GRASP_ELEM_BASE}"
echo "NUM_STEPS_DPO=${NUM_STEPS_DPO}"

# HF_TOKEN / W&B: export in your shell before running; sbatch --export=ALL forwards them.
MASK_JID="$(sbatch --parsable \
  --partition="${MASK_SLURM_PARTITION}" \
  --gres="${MASK_SLURM_GRES}" \
  --export=ALL,"SCRATCH_USER_ROOT=${SCRATCH_USER_ROOT},MASK_RUN_ID=${MASK_RUN_ID}" \
  "${REPO_ROOT}/scripts/run_light_r1_grasp_elem_base_mask.slurm")"

echo ""
echo "Mask job:  ${MASK_JID}"
echo "Mask log:  ${REPO_ROOT}/logs/run_lr1_grasp_elem_base_mask_${MASK_JID}.out"

TRAIN_JID="$(sbatch --parsable \
  --dependency="afterok:${MASK_JID}" \
  --partition="${SPARSE_SLURM_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="${SPARSE_SLURM_GRES}" \
  --mem=128G \
  --time=07:45:00 \
  --job-name="lr1_dpo500_grsp" \
  --export=ALL,"PIPELINE_RUN_ID=${PIPELINE_RUN_ID},RUN_ID=${RUN_ID},PIPELINE_MASK_FILE=${MASK_GRASP_ELEM_BASE},DPO_DATASET_KEY=${DPO_DATASET_KEY},NUM_STEPS_DPO=${NUM_STEPS_DPO},DPO_SAVE_STEPS=${DPO_SAVE_STEPS},DPO_SAVE_TOTAL_LIMIT=${DPO_SAVE_TOTAL_LIMIT}" \
  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh")"

echo "Train job: ${TRAIN_JID}"
echo "Train log: ${REPO_ROOT}/logs/pipeline_sparse_one_mask_${TRAIN_JID}.out"
echo ""
echo "sacct: sacct -j ${MASK_JID},${TRAIN_JID} --format=JobID,State,ExitCode,Elapsed -P"

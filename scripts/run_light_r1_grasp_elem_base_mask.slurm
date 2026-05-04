#!/usr/bin/env bash
# GPU job: Light-R1 GraSP element/base only ("vanilla" — no --score-snr*).
# Same CLI as scripts/orchestrate_light_r1_grasp_sweep.slurm gen_grasp_mask(..., element, use_snr=0).
#
# Logs always under repo logs/ (Slurm directives — same pattern as run_dpo_grasp_masks_only.slurm).
#
# Submit from repo root (optionally pin scratch + run id from login):
#   sbatch --export=ALL,SCRATCH_USER_ROOT=/scratch/biggs.s,MASK_RUN_ID=my_run_001 \
#     scripts/run_light_r1_grasp_elem_base_mask.slurm
#
# Explorer: use multigpu when gpu is drained (matches scripts/sbatch_grpo_llama31_mask_interpretation_suite.sh).
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:15:00
#SBATCH --job-name=lr1_grsp_mk
#SBATCH --output=logs/run_lr1_grasp_elem_base_mask_%j.out
#SBATCH --error=logs/run_lr1_grasp_elem_base_mask_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Same order as scripts/grpo_openr1_llama31_slurm.sh: scratch first, then conda paths.
# sbatch --export=ALL often forwards a broken TRAIN_ENV from an earlier login session
# (e.g. /conda_envs/rl_casino when SCRATCH was empty); never trust it unless the binary exists.
SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
if [[ -z "${SCRATCH_USER_ROOT// }" ]]; then
  SCRATCH_USER_ROOT="/scratch/${USER:-unknown}"
fi
export SCRATCH_USER_ROOT
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"

# Poisoned login env produced MASK_OUT_BASE=/rl_casino_masks — reset.
if [ -z "${MASK_OUT_BASE:-}" ] || [[ "${MASK_OUT_BASE}" == /rl_casino_* ]]; then
  echo "NOTE: resetting MASK_OUT_BASE to ${SCRATCH_USER_ROOT}/rl_casino_masks (was: ${MASK_OUT_BASE:-<unset>})" >&2
  MASK_OUT_BASE="${SCRATCH_USER_ROOT}/rl_casino_masks"
fi
export MASK_OUT_BASE

_default_train_env="${SCRATCH_USER_ROOT}/conda_envs/rl_casino"
if [ -n "${TRAIN_ENV:-}" ] && [ -x "${TRAIN_ENV}/bin/python" ]; then
  :
else
  TRAIN_ENV="${_default_train_env}"
fi
TRAIN_PY="${TRAIN_ENV}/bin/python"
export TRAIN_ENV
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export GRASP_OBJECTIVE="${GRASP_OBJECTIVE:-dpo_preference}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export COLD_CAV_SUBSET="${COLD_CAV_SUBSET:-512}"
export COLD_CAV_NUM_BATCHES="${COLD_CAV_NUM_BATCHES:-32}"

export MASK_RUN_ID="${MASK_RUN_ID:-lr1_grasp_elem_base_${SLURM_JOB_ID:-manual}}"
MASK_DIR="${MASK_OUT_BASE}/${MASK_RUN_ID}"
mkdir -p "${MASK_DIR}"

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
  exit 1
fi

"$TRAIN_PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in this job; masks require a GPU allocation.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: device_count={torch.cuda.device_count()} device0={torch.cuda.get_device_name(0)!r}")
PY

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

MODEL_SANITIZED="$(sanitize_model_name "$MODEL")"
# Filename matches orchestrate_light_r1_grasp_sweep.slurm (MASK_GRASP_ELEM_BASE).
MASK_GRASP_ELEM_BASE="${MASK_DIR}/grasp_${MODEL_SANITIZED}_light_r1_sp${SPARSITY_PERCENT}pct_obj${GRASP_OBJECTIVE}_elem_base.pt"

echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "SCRATCH_USER_ROOT=${SCRATCH_USER_ROOT}"
echo "MASK_RUN_ID=${MASK_RUN_ID}"
echo "MASK_GRASP_ELEM_BASE=${MASK_GRASP_ELEM_BASE}"

echo ""
echo "=== GraSP Light-R1 element/base → ${MASK_GRASP_ELEM_BASE}"
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --method grasp \
  --mode dpo \
  --dataset_name "${DPO_DS_KEY_LIGHT_R1}" \
  --grasp-objective "${GRASP_OBJECTIVE}" \
  --n_samples "${COLD_CAV_SUBSET}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --batch_size 1 \
  --max_length "${DPO_MAX_LENGTH}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  --grasp-preference-beta "${DPO_BETA}" \
  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
  --output "${MASK_GRASP_ELEM_BASE}"

ls -la "${MASK_GRASP_ELEM_BASE}"
echo "=== Done ==="

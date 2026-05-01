#!/usr/bin/env bash
# GPU job: build ONLY GraSP element/SNR DPO masks for Tulu3 + Light-R1 (.pt files).
# Same paths / CLI as scripts/orchestrate_dpo_gm_sweep.slurm `build_grasp_dpo_mask`, without
# magnitude masks or sparse training submits. Use if GraSP fails mid-sweep or you want masks alone.
#
# Submit from repo root:
#   sbatch scripts/run_dpo_grasp_masks_only.slurm
#
# With exports:
#   sbatch --export=ALL,SCRATCH_USER_ROOT=/scratch/$USER,HF_TOKEN="$HF_TOKEN" \
#     scripts/run_dpo_grasp_masks_only.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpo_grasp_mk
#SBATCH --output=logs/run_dpo_grasp_masks_%j.out
#SBATCH --error=logs/run_dpo_grasp_masks_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export MASK_OUT_BASE="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

export DPO_DS_KEY_TULU3="${DPO_DS_KEY_TULU3:-tulu3}"
export DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"
export DPO_DS_KEY_TULU3_SANITIZED="${DPO_DS_KEY_TULU3_SANITIZED:-tulu3}"
export DPO_DS_KEY_LIGHT_R1_SANITIZED="${DPO_DS_KEY_LIGHT_R1_SANITIZED:-light_r1}"

export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export DPO_GRASP_OBJECTIVE="${DPO_GRASP_OBJECTIVE:-dpo_preference}"
export DPO_BETA="${DPO_BETA:-0.1}"
export COLD_CAV_SUBSET="${COLD_CAV_SUBSET:-512}"
export COLD_CAV_NUM_BATCHES="${COLD_CAV_NUM_BATCHES:-32}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-1}"

export ORCH_SCORE_SNR_MODE="${ORCH_SCORE_SNR_MODE:-per_weight}"
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-48}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

EXTRA_GRASP_FLAGS="${EXTRA_GRASP_FLAGS:-}"
export MASK_RUN_ID="${MASK_RUN_ID:-dpo_grasp_only_${SLURM_JOB_ID:-manual}}"
MASK_DIR="${MASK_OUT_BASE}/${MASK_RUN_ID}"
mkdir -p "${MASK_DIR}"

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi

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

"$TRAIN_PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: need CUDA", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK {torch.cuda.get_device_name(0)!r}")
PY

SNR_ARGS=( --score-snr "${ORCH_SCORE_SNR_MODE}"
           --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}"
           --score-snr-eps "${ORCH_SCORE_SNR_EPS}"
           --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}" )
if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
  SNR_ARGS+=( --score-snr-allow-large-ram )
fi

run_one_grasp() {
  local ds_key="$1"
  local ds_sanitized="$2"
  local label="$3"
  local grasp_out="${MASK_DIR}/grasp_${MODEL_SANITIZED}_${ds_sanitized}_sp${SPARSITY_PERCENT}pct_obj${DPO_GRASP_OBJECTIVE}_elem_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}.pt"

  echo ""
  echo "=== GraSP (${label}) → ${grasp_out}"
  "$TRAIN_PY" src/cold_start/inference_mask_finder.py \
    --model_name "${MODEL}" \
    --method grasp \
    --mode dpo \
    --dataset_name "${ds_key}" \
    --grasp-objective "${DPO_GRASP_OBJECTIVE}" \
    --n_samples "${COLD_CAV_SUBSET}" \
    --sparsity "${SPARSITY_PERCENT}" \
    --batch_size "${ORCH_MASK_BATCH_SIZE}" \
    --max_length "${DPO_MAX_LENGTH}" \
    --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
    --grasp-preference-beta "${DPO_BETA}" \
    --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
    "${SNR_ARGS[@]}" \
    --mask-granularity element \
    --output "${grasp_out}" \
    ${EXTRA_GRASP_FLAGS}
  ls -la "${grasp_out}"
}

FAILED=0
run_one_grasp "${DPO_DS_KEY_TULU3}" "${DPO_DS_KEY_TULU3_SANITIZED}" "Tulu3" || FAILED=1
run_one_grasp "${DPO_DS_KEY_LIGHT_R1}" "${DPO_DS_KEY_LIGHT_R1_SANITIZED}" "Light-R1" || FAILED=1

echo ""
echo "Mask dir: ${MASK_DIR}"
ls -lh "${MASK_DIR}"

if [ "${FAILED}" -ne 0 ]; then
  echo "ERROR: one or more GraSP builds failed." >&2
  exit 1
fi

echo ""
echo "If these paths match orchestrate_dpo_gm_sweep, you can point PIPELINE_MASK_FILE at them and"
echo "sbatch scripts/pipeline_sparse_one_mask.sh, or re-run orchestrate_dpo_gm_sweep once magnitude"
echo "masks already exist (it will regenerate GraSP faster after code fix) or invoke submit steps manually."

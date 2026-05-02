#!/usr/bin/env bash
# GPU job: build TWO Open-R1 GRPO GraSP masks on one allocation (mask-only; no training submit):
#   1) Vanilla GraSP-ABS  (--score-snr off, --grasp-rank-direction abs)
#   2) SNR + GraSP-ABS    (--score-snr per_weight + log1p, --grasp-rank-direction abs)
#
# Submit from repo root (non-interactive):
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" scripts/run_grpo_grasp_vanilla_vs_snr_abs.slurm
#
# Optional overrides (same env names as run_grpo_grasp_mask_only.slurm):
#   MODEL, GRPO_DATASET_HF, SPARSITY_PERCENT, GRPO_N_SAMPLES, GRASP_EXTRA_FLAGS, GRPO_MASK_SUBDIR, etc.
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_gr2mask
#SBATCH --output=logs/run_grpo_grasp_vanilla_snr_abs_%j.out
#SBATCH --error=logs/run_grpo_grasp_vanilla_snr_abs_%j.err

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

export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export GRPO_GRASP_OBJECTIVE="${GRPO_GRASP_OBJECTIVE:-lm}"
export GRPO_GRASP_PREFERENCE_BETA="${GRPO_GRASP_PREFERENCE_BETA:-1.0}"
export GRPO_N_SAMPLES="${GRPO_N_SAMPLES:-256}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-1}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"

# SNR branch only (vanilla run ignores these except for output naming clarity).
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-96}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

# Optional extra flags for inference_mask_finder (e.g. --also-emit-rank-variants)
GRASP_EXTRA_FLAGS="${GRASP_EXTRA_FLAGS:-}"

# Optional subdirectory under GRPO_MASK_DIR for this batch (default: job-specific).
GRPO_MASK_SUBDIR="${GRPO_MASK_SUBDIR:-grasp_cmp_${SLURM_JOB_ID:-manual}}"
MASK_SUBDIR_FULL="${GRPO_MASK_DIR}/${GRPO_MASK_SUBDIR}"
mkdir -p "${MASK_SUBDIR_FULL}"

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
DS_TAIL="${GRPO_DATASET_HF##*/}"
SP_TAG="$(echo "${SPARSITY_PERCENT}" | tr '.' '_')"

BASE_STEM="grpo_grasp_${GRPO_GRASP_OBJECTIVE}_sp${SP_TAG}_${MODEL_SANITIZED}_${DS_TAIL}_elem_rankabs"
OUT_VANILLA="${MASK_SUBDIR_FULL}/${BASE_STEM}_snroff.pt"
OUT_SNR_ABS="${MASK_SUBDIR_FULL}/${BASE_STEM}_snrperweight_${ORCH_SCORE_SNR_TRANSFORM}.pt"

echo "REPO_ROOT=${REPO_ROOT}"
echo "MASK_SUBDIR_FULL=${MASK_SUBDIR_FULL}"
echo "Vanilla ABS → ${OUT_VANILLA}"
echo "SNR+ABS     → ${OUT_SNR_ABS}"

"$TRAIN_PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: need CUDA", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK {torch.cuda.get_device_name(0)!r}")
PY

common_grasp_args=(
  --model_name "${MODEL}"
  --mode grpo
  --method grasp
  --grasp-objective "${GRPO_GRASP_OBJECTIVE}"
  --dataset_name "${GRPO_DATASET_HF}"
  --n_samples "${GRPO_N_SAMPLES}"
  --sparsity "${SPARSITY_PERCENT}"
  --max_length "${GRPO_MAX_PROMPT_LENGTH}"
  --batch_size "${ORCH_MASK_BATCH_SIZE}"
  --seed 42
  --grasp-preference-beta "${GRPO_GRASP_PREFERENCE_BETA}"
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}"
  --grasp-rank-direction abs
  --mask-granularity element
)

echo ""
echo "=== [1/2] Vanilla GraSP (SNR off) + ABS ranking ==="
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  "${common_grasp_args[@]}" \
  --output "${OUT_VANILLA}" \
  ${GRASP_EXTRA_FLAGS}

echo ""
echo "=== [2/2] GraSP + per-weight SNR + ABS ranking ==="
SNR_ARGS=(
  --score-snr per_weight
  --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}"
  --score-snr-eps "${ORCH_SCORE_SNR_EPS}"
  --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}"
)
if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
  SNR_ARGS+=( --score-snr-allow-large-ram )
fi

"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  "${common_grasp_args[@]}" \
  "${SNR_ARGS[@]}" \
  --output "${OUT_SNR_ABS}" \
  ${GRASP_EXTRA_FLAGS}

echo ""
echo "Done. Outputs:"
ls -la "${OUT_VANILLA}" "${OUT_SNR_ABS}"

#!/usr/bin/env bash
# E1: DPO gradient energy captured by each Llama-3.1-8B mask (phi = ||g⊙M|| / ||g||).
#
# One DPO backward at the pretrained init over 64 × 2 = 128 Tülu3 examples — the same effective
# batch the compared runs take their first optimizer step on — then phi for all eight masks
# against that single gradient. Every mask here is Llama-3.1-8B-shaped, DPO-derived and
# GRPO-derived alike, so one gradient serves all of them.
#
# The GRPO masks are the high-value arm and cost nothing extra: scoring GRPO-selected coordinates
# against a DPO gradient tests whether they carry any DPO signal at all, which is a measured
# mechanism for the cross-objective transfer gap in Figure 3.
#
# Run from repo root:
#   sbatch scripts/e1_grad_energy_llama31_tulu3.slurm
#
# Smoke test first (small model, synthetic uniform mask, expect phi ≈ sqrt(density)):
#   E1_SMOKE=1 sbatch scripts/e1_grad_energy_llama31_tulu3.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=e1_grad_energy
#SBATCH --output=logs/e1_grad_energy_%j.out
#SBATCH --error=logs/e1_grad_energy_%j.err
#SBATCH --export=ALL

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
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY} (set TRAIN_ENV)" >&2
  exit 1
fi

export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Pure analysis job: no run to log, and pipeline_common would force WANDB_MODE=online.
export WANDB_MODE=disabled
export HF_HOME="${HF_HOME:-${SCRATCH_USER_ROOT}/hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
_HF_DEFAULT_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${_HF_DEFAULT_TOKEN_FILE}" ]; then
  export HF_TOKEN="$(tr -d '\n\r' < "${_HF_DEFAULT_TOKEN_FILE}")"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

OUT_DIR="${OUT_DIR:-${SCRATCH_USER_ROOT}/rl_casino_analysis/rebuttal_e1_e2/e1}"
MICROBATCHES="${MICROBATCHES:-64}"
MASK_ROOT="${MASK_ROOT:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
GRPO_MASK_ROOT="${GRPO_MASK_ROOT:-${SCRATCH_USER_ROOT}/rl_casino_grpo/masks/evolstudy}"
TULU3_MASK_DIR="${MASK_ROOT}/tulu3_mag_oracle_rand_dpo5k_dense_tulu3"

if [ "${E1_SMOKE:-0}" = "1" ]; then
  # Instrument check only: 0.6B model, 2 microbatches, synthetic uniform mask at 2.5%.
  # A synthetic mask has no per-layer floor, so both chance baselines should sit near
  # sqrt(0.025) = 0.1581 and phi should land on them.
  "$TRAIN_PY" src/analysis/grad_energy_capture.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset tulu3 \
    --microbatches 2 \
    --synthetic_mask_density 0.025 \
    --out_dir "${OUT_DIR}_smoke"
  exit 0
fi

ARGS=(
  src/analysis/grad_energy_capture.py
  --model_name meta-llama/Llama-3.1-8B-Instruct
  --dataset tulu3
  --microbatches "${MICROBATCHES}"
  --per_device_train_batch_size 2
  --max_length 1024
  --max_prompt_length 512
  --beta 0.1
  --seed 42
  --out_dir "${OUT_DIR}"
  # DPO / Tülu3 masks, ρ=97.5%
  --mask "oracle_dpo_tulu3=${TULU3_MASK_DIR}/checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct.pt"
  --mask "warm_dpo_tulu3_k200=${TULU3_MASK_DIR}/warm_magnitude_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200.pt"
  --mask "random_dpo_tulu3=${TULU3_MASK_DIR}/random_baseline_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200_seed42.pt"
  # GRPO / Open-R1 masks, ρ=97.5% — scored against the same DPO gradient
  --mask "oracle_grpo_openr1=${GRPO_MASK_ROOT}/oracle_gt.pt"
  --mask "warm_grpo_k50=${GRPO_MASK_ROOT}/mag_step50.pt"
  --mask "warm_grpo_k100=${GRPO_MASK_ROOT}/mag_step100.pt"
  --mask "warm_grpo_k250=${GRPO_MASK_ROOT}/mag_step250.pt"
  --mask "random_grpo=${GRPO_MASK_ROOT}/random.pt"
)

echo "OUT_DIR=${OUT_DIR}"
"$TRAIN_PY" "${ARGS[@]}"

echo ""
echo "===== E1 done ====="
echo "  ${OUT_DIR}/e1_grad_energy.json"
echo "  ${OUT_DIR}/e1_per_tensor.csv"
echo ""
echo "VALIDITY GATE: phi(random_dpo_tulu3) and phi(random_grpo) must sit near their"
echo "chance_energy_weighted values. If they do not, the instrument is wrong — debug before"
echo "reporting any phi."

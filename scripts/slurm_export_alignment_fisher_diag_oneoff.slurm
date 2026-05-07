#!/usr/bin/env bash
# One-off Fisher-diagonal export for alignment optimal scores.
#
# Uses the "multigpu partition + request 1 GPU" trick.
# Run from repo root (contains scripts/ and src/):
#   export CKPT500=/scratch/$USER/rl_casino_train/dpo5k_dense_light-r1/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500
#   export FISHER_OUT=/scratch/$USER/rl_casino_analysis/alignment/fisher_diag_raw_ckpt500.pt
#   sbatch scripts/slurm_export_alignment_fisher_diag_oneoff.slurm
#
# Optional knobs:
#   export FISHER_SAMPLES=256 FISHER_MAX_LEN=512 FISHER_MINI_BS=4
#   export FISHER_DATASET=qihoo360/Light-R1-DPOData
#   export FISHER_MLP_ONLY=1
#
# Outputs:
#   $FISHER_OUT (.pt dict name->tensor float32)
#   ${FISHER_OUT%.pt}.json metadata

#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=fisher_diag
#SBATCH --output=logs/fisher_diag_%j.out
#SBATCH --error=logs/fisher_diag_%j.err
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

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CKPT500="${CKPT500:-}"
if [ -z "${CKPT500}" ]; then
  echo "ERROR: set CKPT500=/scratch/$USER/.../checkpoint-500" >&2
  exit 1
fi

FISHER_OUT="${FISHER_OUT:-${SCRATCH_USER_ROOT}/rl_casino_analysis/alignment/fisher_diag_raw_ckpt500.pt}"
FISHER_SAMPLES="${FISHER_SAMPLES:-512}"
FISHER_MAX_LEN="${FISHER_MAX_LEN:-512}"
FISHER_MINI_BS="${FISHER_MINI_BS:-4}"
FISHER_DATASET="${FISHER_DATASET:-qihoo360/Light-R1-DPOData}"

mkdir -p "$(dirname "${FISHER_OUT}")"

ARGS=(scripts/export_alignment_fisher_diag.py
  --model "${CKPT500}"
  --dataset "${FISHER_DATASET}"
  --n-samples "${FISHER_SAMPLES}"
  --max-length "${FISHER_MAX_LEN}"
  --mini-batch-size "${FISHER_MINI_BS}"
  --out "${FISHER_OUT}"
)
if [ "${FISHER_MLP_ONLY:-0}" = "1" ]; then
  ARGS+=(--mlp-only)
fi

echo "Running: ${TRAIN_PY} ${ARGS[*]}"
"${TRAIN_PY}" "${ARGS[@]}"

echo "Done. Fisher diag: ${FISHER_OUT}"


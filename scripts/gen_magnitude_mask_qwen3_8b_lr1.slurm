#!/bin/bash
# Generate warm magnitude mask for Qwen3-8B / light-r1 from step-250 float32 delta files.
#
# Submit from repo root:
#   sbatch scripts/gen_magnitude_mask_qwen3_8b_lr1.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=mag_mask_q38b
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mag_mask_qwen3_8b_%j.out
#SBATCH --error=logs/mag_mask_qwen3_8b_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

DELTA_DIR="/scratch/biggs.s/rl_casino_train/20260601_175443_7366081/deltas/qwen_qwen3_8b_light_r1"
OUTPUT_FILE="/scratch/biggs.s/rl_casino_masks/warm_magnitude_qwen3_8b_light_r1_sp97.5pct_step250.pt"

TRAIN_ENV="${TRAIN_ENV:-/scratch/biggs.s/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"

mkdir -p "$(dirname "$OUTPUT_FILE")" logs

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Delta dir : ${DELTA_DIR}"
echo "Output    : ${OUTPUT_FILE}"
ls "${DELTA_DIR}/"

"$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir    "$DELTA_DIR" \
  --method           magnitude \
  --sparsity_percent 97.5 \
  --target_step      250 \
  --min_layer_keep_ratio 0.0025 \
  --output_file      "$OUTPUT_FILE"

echo "Done: ${OUTPUT_FILE}"

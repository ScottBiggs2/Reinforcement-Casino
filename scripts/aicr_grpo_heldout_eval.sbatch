#!/bin/bash
# Held-out GSM8K eval for one matched-schedule GRPO arm.
#
# Turns the training-curve result into a capability claim. Measurement is identical
# to training (same prompt suffix, same accuracy_reward extraction); only the data is
# held out. GSM8K test is used because it cannot be contaminated -- see the module
# docstring in src/evaluation/grpo_heldout_eval.py.
#
# Usage:
#   MODEL_PATH=<ckpt-500 dir> TAG=oracle97.5 sbatch aicr_grpo_heldout_eval.sbatch
#SBATCH --job-name=grpo_eval
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=06:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/grpo_eval_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

: "${MODEL_PATH:?Set MODEL_PATH=<checkpoint dir or HF id>}"
: "${TAG:?Set TAG=dense|oracle97.5|random|rho99|rho99.75|base}"
OUT_DIR="${OUT_DIR:-$S/rebuttal_analysis/grpo_heldout_eval}"
mkdir -p "$OUT_DIR"

[ -f "$OUT_DIR/$TAG.json" ] && { echo "already evaluated: $OUT_DIR/$TAG.json"; exit 0; }

echo "=========================================="
echo "Held-out GSM8K eval | arm=$TAG"
echo "  model=$MODEL_PATH"
echo "  greedy, max_new_tokens=2048, scorer=training accuracy_reward"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

# Invoked as a file, not `-m`: src/evaluation/__init__.py does `from evaluation.…`,
# which only resolves when src/ itself is the path root. Running the file directly
# skips that package init while PYTHONPATH=repo-root keeps `src.utils…` importable.
python src/evaluation/grpo_heldout_eval.py \
    --model_path "$MODEL_PATH" \
    --tag "$TAG" \
    --out_dir "$OUT_DIR" \
    --batch_size "${BATCH_SIZE:-32}"

echo "Finished: $(date)"

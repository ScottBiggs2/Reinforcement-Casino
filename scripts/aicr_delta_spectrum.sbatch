#!/bin/bash
# Full-depth delta analysis with exact singular spectra, for the LoRA-rebuttal
# geometry figures (Nadim, 2026-07-31).
#
# The 2026-07-25 Explorer run (job 8725606) established stable rank >> 64 but had
# two gaps this closes:
#   1. rank_max_tensors=48 sampled only embed + layers 0-6. Here all 226 matched
#      2-D tensors are ranked, so the figure covers full depth.
#   2. Stable rank only bounds the best-rank-r capture from above (Eckart-Young:
#      <= r/srank). --spectrum_topk 256 records the top-256 sigma_i^2 per matrix,
#      so the figure can show the EXACT fraction of ||dW||^2 any rank-64 update
#      can represent, next to the measured top-2.5%-coordinate capture.
#   3. The Explorer trio had no Llama DPO Light-R1 arm (its dense ckpt was deleted
#      pre-retrain). Arm 0 is that run — the same task the LoRA r=64 baseline
#      trained on, so the comparison becomes task-matched.
#
# Pure CPU tensor work — cpu partition, no GPU. Peak memory is one fp32 8B state
# dict + one streamed model (~70 GB); randomized SVD adds one fp32 tensor copy.
#
# Usage: sbatch scripts/aicr_delta_spectrum.sbatch
#SBATCH --job-name=delta_spectrum
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/delta_spectrum_%A_%a.out

set -uo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

LLAMA="meta-llama/Llama-3.1-8B-Instruct"

# tag | initial (theta^0) | final (theta_T)
CFG=(
  "llama_dpo_lightr1|$LLAMA|$S/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500"
  "llama_grpo_math220k|$LLAMA|$S/transfer_v1/dense_grpo_math220k_llama8b/checkpoints/checkpoint-500"
)
IFS='|' read -r TAG INIT FINAL <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"

OUT_DIR="$S/rebuttal_analysis"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "delta spectrum: $TAG"
echo "  theta^0 : $INIT"
echo "  theta_T : $FINAL"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

if [ ! -d "$FINAL" ]; then echo "FAILED: missing $FINAL"; exit 1; fi

python src/analysis/delta_analysis.py \
    --initial "$INIT" \
    --final "$FINAL" \
    --fractions 0.0025,0.01,0.025,0.05,0.10,0.25 \
    --stable_rank \
    --rank_max_tensors 226 \
    --spectrum_topk 256 \
    --tag "${TAG}_spec" \
    --out "$OUT_DIR/delta_${TAG}_spec.json"
RC=$?
if [ "$RC" -ne 0 ]; then echo "FAILED $TAG rc=$RC: $(date)"; exit "$RC"; fi
echo "Finished $TAG: $(date)"

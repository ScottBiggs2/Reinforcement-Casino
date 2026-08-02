#!/bin/bash
# Oracle mask for the GRPO Tülu3-RLVR dense run — checkpoint-diff route, the same
# route that built the paper's oracle_grpo_math220k_step500_sp97.5.pt (2026-04-29),
# so the Figure-2 GRPO⇄GRPO pair compares masks of identical construction.
#
# Chained: submit with --dependency=afterok:<dense job 239122>.
# CPU/memory-latency bound; measured checkpoint-diff peak at rho=97.5 was 105 GB.
#
# Usage: sbatch --dependency=afterok:239122 aicr_mask_grpo_tulu3rlvr.sbatch
#SBATCH --job-name=t3rlvr_mask
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/t3rlvr_mask_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

CKPT="$S/rebuttal_analysis/grpo_matched_aicr/dense_tulu3rlvr/grpo_matched_dense_tulu3rlvr_500steps_cap2048/checkpoints/checkpoint-500"
OUT="$S/transfer_v1/oracle_masks_llama8b/oracle_grpo_tulu3rlvr_step500_sp97.5.pt"

[ -d "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
[ -f "$OUT" ] && { echo "Mask already exists: $OUT"; exit 0; }

echo "=========================================="
echo "GRPO Tülu3-RLVR oracle mask | checkpoint-diff | rho=97.5"
echo "  ckpt=$CKPT"
echo "  out=$OUT"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

cd /work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model meta-llama/Llama-3.1-8B-Instruct \
    --final_model "$CKPT" \
    --sparsity_percent 97.5 \
    --force_cpu \
    --output_file "$OUT"

echo "Finished: $(date)"
python - "$OUT" <<'PY'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
masks = m.get("masks", m)
kept = sum(v.sum().item() for v in masks.values())
tot = sum(v.numel() for v in masks.values())
print(f"kept {kept:,} / {tot:,} = {kept/tot:.4%} (target 2.5%)")
PY

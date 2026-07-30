#!/bin/bash
# GRPO oracle-mask rho sweep — mask generation (AICR cpu partition).
#
# Levels chosen from the measured GRPO delta concentration curve
# (delta_llama_grpo_math220k.json): keep-5% captures 99.93% of update mass,
# keep-1% captures 80.9%, keep-0.25% captures 48.0%. Together with the running
# rho=97.5 arm (96.4%) this spans saturation -> breakage. rho<=90 skipped:
# mass is 100% there, success would be uninformative (DO_NOT_REPEAT.md).
#
# Source = the SAME trajectory as the existing 97.5 oracle (paper dense GRPO
# checkpoint-500, transferred from Explorer 2026-07-30, md5-verified), same
# checkpoint-diff code and conventions — so against the 97.5 arm the only
# variable is rho.
#
# Memory per level tracks keep count (measured Tulu3 table: 0.20B->105G,
# 0.08B->98G): rho95 keeps 0.40B -> 180G ask; 99/99.75 -> 120G.
#SBATCH --job-name=grpo_rho_masks
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/grpo_rho_masks_%A_%a.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

CKPT="$S/transfer_v1/dense_grpo_math220k_llama8b/checkpoints/checkpoint-500"
OUT_DIR="$S/transfer_v1/oracle_masks_llama8b"
mkdir -p "$OUT_DIR"

case "${SLURM_ARRAY_TASK_ID}" in
  0) RHO=95    ;;
  1) RHO=99    ;;
  2) RHO=99.75 ;;
  *) echo "FATAL: unknown array id"; exit 1 ;;
esac
OUT="$OUT_DIR/oracle_grpo_math220k_step500_sp${RHO}.pt"

[ -f "$CKPT/model.safetensors.index.json" ] || { echo "FATAL: ckpt-500 not staged"; exit 1; }
[ -f "$OUT" ] && { echo "mask already exists: $OUT"; exit 0; }

echo "rho=${RHO} -> $OUT  ($(date))"
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model meta-llama/Llama-3.1-8B-Instruct \
    --final_model "$CKPT" \
    --sparsity_percent "$RHO" \
    --output_file "$OUT" \
    --force_cpu
RC=$?
ls -la "$OUT"
exit $RC

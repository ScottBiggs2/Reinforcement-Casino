#!/bin/bash
# GRPO oracle rho sweep — LOW-sparsity levels {90, 80, 70} (plateau region).
#
# Complement to aicr_grpo_rho_sweep_masks.sbatch ({95,99,99.75}, cliff region).
# These levels sit at 100% measured mass capture, so the mechanism prediction is
# a flat plateau at ~dense performance; their value is (a) answering Mukherjee's
# 70% natural-sparsity anchor with data rather than argument, and (b) grid
# parity with the DPO sweep {80,90,97.5,99}. Requested by Irene 2026-07-29.
#
# Same source ckpt, code, and conventions as all other GRPO oracles.
# Memory tracks keep count (Tulu3 measured: 0.80B->161G, 1.61B->199G;
# 70 keeps 2.41B, extrapolated ~300G) — pass --mem at submit time per level:
#   RHO=90 sbatch --mem=200G --time=03:00:00 ...
#   RHO=80 sbatch --mem=240G --time=05:00:00 ...
#   RHO=70 sbatch --mem=320G --time=07:00:00 ...
#SBATCH --job-name=grpo_rho_masks_lo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/grpo_rho_masks_lo_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

: "${RHO:?Set RHO=90|80|70}"
CKPT="$S/transfer_v1/dense_grpo_math220k_llama8b/checkpoints/checkpoint-500"
OUT="$S/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp${RHO}.pt"

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

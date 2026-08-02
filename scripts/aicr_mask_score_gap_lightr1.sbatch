#!/bin/bash
# Recompute the Figure 5 certifiability-margin data (mask_score_gap_histograms.npz)
# on AICR — the original NPZ did not survive the 2026-04-26 cleanup on Explorer.
#
# Inputs are the migrated dense DPO Light-R1 run under transfer_v1: base model from
# HF cache, checkpoint-500 as the oracle endpoint, and warm-start delta snapshots at
# steps 50/100/150/200 (deltas_step_500.pt doubles as the verify set). Settings match
# the submitted figure: rho=97.5, global tau (Theorem-3 top-k), tie-break matched.
#
# Pure CPU tensor work — cpu partition, no GPU (no idle-GPU canceller on AICR, and
# nothing here would keep one busy).
#
# Usage: sbatch aicr_mask_score_gap_lightr1.sbatch
#SBATCH --job-name=mask_score_gap
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/mask_score_gap_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

RUN_ROOT=/scratch/xie_yiyi_neu/transfer_v1/dense_dpo_lightr1_llama8b
CKPT500="$RUN_ROOT/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500"
DELTAS="$RUN_ROOT/deltas/meta_llama_llama_3_1_8b_instruct_light_r1"
OUT_DIR="$S/rebuttal_analysis/mask_score_gap_lightr1/run_global_tau_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

export CHECKPOINT_DTYPE=bfloat16
export CERT_GLOBAL_MODE=stream

echo "=========================================="
echo "Figure 5 recompute — mask score gap | DPO Light-R1 | rho=97.5 | global tau"
echo "  ckpt500=$CKPT500"
echo "  deltas=$DELTAS"
echo "  out=$OUT_DIR"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

python src/analysis/mask_score_gap_analysis.py \
    --initial_model meta-llama/Llama-3.1-8B-Instruct \
    --final_model "$CKPT500" \
    --delta_log_dir "$DELTAS" \
    --magnitude_milestones 50,100,150,200 \
    --random_seed 42 \
    --sparsity_percent 97.5 \
    --cert_tau_rule global \
    --cert_global_mode stream \
    --verify_deltas_dir "$DELTAS" \
    --out_dir "$OUT_DIR"

echo "Finished: $(date)"
ls -la "$OUT_DIR"

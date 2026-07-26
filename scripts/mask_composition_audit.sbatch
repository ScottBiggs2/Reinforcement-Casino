#!/bin/bash
# What is the 2.5% subnetwork actually made of? Tensor-class composition audit.
#
# CPU-only: no model forward, no GPU. Uses the `short` partition (18 idle nodes,
# 2-day limit) rather than borrowing a GPU, so there is no idle-GPU-cancel exposure.
#
# Three questions, all of which change how the submitted results should be read:
#
#  1. Is the subnetwork mostly LayerNorm + bias? checkpoint_diff_mask_finder.py has no
#     exclusion list, and LN/bias move with large RELATIVE magnitude, so any |dtheta|
#     scorer over-selects them. BitFit (Ben Zaken et al., ACL 2022) already recovers
#     most of full fine-tuning from biases alone (~0.08% of params) -- if that is what
#     the oracle found, the finding is a known one.
#
#  2. Do oracle and random spend their budget on the SAME tensor classes? The random
#     baseline gets a per-layer keep floor (min_layer_keep_ratio, default 2.5e-3), which
#     would hand it the LN/bias tensors too. If the budget-share columns agree, then
#     "oracle vs random" was never testing the scoring function -- which would EXPLAIN
#     the oracle ~= random pattern that three reviewers independently flagged.
#
#  3. Do the DPO and GRPO oracles differ in composition? The paper's central claim is
#     that structure is set by the objective. Jaccard says the two masks pick different
#     coordinates; composition says whether they pick different KINDS of tensors. That
#     is a second, independent axis for the same claim, at zero training cost.
#
# The anti-oracle (bottom-2.5% by |dtheta|) is included because the mask already exists
# and has never been analysed -- it is the natural lower bound for every column.
#SBATCH --job-name=mask_composition
#SBATCH --output=logs/mask_composition_%j.out
#SBATCH --error=logs/mask_composition_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO"
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

M="/scratch/$USER/transfer_v1/oracle_masks_llama8b"
OUT="/scratch/$USER/rebuttal_analysis"
mkdir -p "$OUT"

echo "=========================================="
echo "Mask composition audit — Llama-3.1-8B masks @ 97.5%"
echo "Job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

# Load one at a time (each is ~8 GB of bool); audit() scopes each mask locally.
python src/analysis/mask_composition.py \
    "$M/oracle_dpo_lightr1_step500_sp97.5.pt" \
    "$M/random_baseline_lightr1_sp97.5_seed42.pt" \
    "$M/anti_oracle_dpo_lightr1_step500_sp97.5.pt" \
    "$M/oracle_dpo_tulu3_step500_sp97.5.pt" \
    "$M/oracle_grpo_math220k_step500_sp97.5.pt" \
    --out "$OUT/mask_composition_llama8b.json"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Report: $OUT/mask_composition_llama8b.json"
echo "=========================================="

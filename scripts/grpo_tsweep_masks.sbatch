#!/bin/bash
# Build the GRPO magnitude/oracle mask family from ONE dense trajectory.
#
#   T = 50, 100, 150, 200, 250  ->  warm-start magnitude masks  |theta^T - theta^0|
#   T = 500                     ->  the oracle (k = T), same formula at end of training
#
# 150 and 200 are not free additions of convenience: k = 50/100/150/200 is exactly the
# warm-magnitude grid the submission uses for DPO (Fig. 1, Fig. 5), and 200 is the
# `magnitude.target_step` locked in Scott's docs/hyperparams/grpo_500step_5way_sweep.yaml.
# Including them makes the GRPO sweep line up point-for-point with the DPO one instead
# of being a differently-sampled curve that has to be argued about.
#
# All four come from the same run (scripts/grpo_tsweep_dense.sbatch, 500 steps,
# cosine + 50-step warmup, cap 1024), so "how much of the oracle is already visible
# at step T" is a statement about the trajectory and not about a schedule mismatch.
#
# MASK-FINDER VERSION — deliberate choice.
# We use the version on this branch, NOT origin/cav_fixes. Scott's branch has since
# restricted scoring to 2-D `weight` tensors only (excluding 1-D norms) and dropped
# the seed-jitter path. That is arguably the better definition and is what the random
# baseline uses, but it is NOT the code that produced
# `oracle_grpo_math220k_step500_sp97.5.pt` (2026-04-29). Keeping this branch's version
# means the new T=500 oracle is directly comparable to the paper's oracle, which turns
# the pair into a free reproducibility check: same objective, same dataset, same
# sparsity, different LR schedule. Any Jaccard shortfall there is schedule sensitivity.
#
# Defaults left untouched to match the paper's masks: global pooling (no --local_pool),
# DEFAULT_MIN_LAYER_KEEP_RATIO, element-wise granularity.
#
# CPU by design: mask generation spends most of its wall-clock loading two 8B state
# dicts, and RC cancels GPU jobs that sit idle. No GPU is requested, so there is
# nothing to reclaim.
#SBATCH --job-name=grpo_tsweep_masks
#SBATCH --output=logs/grpo_tsweep_masks_%j.out
#SBATCH --error=logs/grpo_tsweep_masks_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO" || exit 1
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_USER_ROOT:-/scratch/$USER}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
SPARSITY="${SPARSITY:-97.5}"
STEPS="${STEPS:-500}"
CKPT_DIR="$SCRATCH_ROOT/rebuttal_analysis/grpo_tsweep/dense/grpo_tsweep_dense_${STEPS}steps/checkpoints"
MASK_DIR="$SCRATCH_ROOT/rebuttal_analysis/grpo_tsweep/masks"
mkdir -p "$MASK_DIR"

TS=(50 100 150 200 250 500)

echo "=========================================="
echo "GRPO T-sweep mask generation"
echo "  base      = $MODEL_BASE"
echo "  sparsity  = ${SPARSITY}%"
echo "  ckpt_dir  = $CKPT_DIR"
echo "  mask_dir  = $MASK_DIR"
echo "  T values  = ${TS[*]}   (T=$STEPS is the oracle)"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

# Fail loudly and early: a missing T silently turns the comparison into a different
# experiment, so check all of them before spending an hour on the first mask.
MISSING=0
for T in "${TS[@]}"; do
    if [ ! -d "$CKPT_DIR/checkpoint-$T" ]; then
        echo "FATAL: missing checkpoint-$T under $CKPT_DIR"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "Available checkpoints:"; ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V
    exit 1
fi

RC_ALL=0
for T in "${TS[@]}"; do
    if [ "$T" -eq "$STEPS" ]; then KIND="oracle"; else KIND="warmmag"; fi
    OUT="$MASK_DIR/grpo_math220k_${KIND}_T${T}_sp${SPARSITY}.pt"
    echo ""
    echo "--- [T=$T / $KIND] -> $OUT"
    if [ -f "$OUT" ]; then echo "    exists, skipping"; continue; fi
    python src/warm_start/checkpoint_diff_mask_finder.py \
        --initial_model "$MODEL_BASE" \
        --final_model   "$CKPT_DIR/checkpoint-$T" \
        --sparsity_percent "$SPARSITY" \
        --force_cpu \
        --output_file "$OUT"
    rc=$?
    echo "    exit=$rc"
    [ $rc -ne 0 ] && RC_ALL=$rc
done

echo ""
echo "=========================================="
ls -lh "$MASK_DIR" 2>&1
echo "exit=$RC_ALL  finished $(date)"
echo "=========================================="
exit $RC_ALL

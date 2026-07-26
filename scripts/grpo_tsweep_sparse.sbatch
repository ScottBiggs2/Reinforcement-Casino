#!/bin/bash
# Sparse GRPO training on the masks built by scripts/grpo_tsweep_masks.sbatch.
#
# Every arm here is schedule-matched to the dense run in scripts/grpo_tsweep_dense.sbatch:
# cosine, 500 steps, 50-step warmup, lr 5e-6, cap 2048. That matters. Appendix D.4's
# dense-vs-sparse GRPO comparison is unreadable precisely because it was not matched —
# dense sat at 99.8% of peak LR at step 500 while every sparse arm had annealed to 0.2%,
# a 449x gap (docs/run_logs/grpo_schedule_confound_2026-07-25.md). These four arms and
# the dense run share one schedule, so a gap between them is attributable to the mask.
#
# ARMS
#   0  oracle  T=500   the mask the paper calls the ground truth, rebuilt schedule-clean
#   1  warmmag T=250   half the trajectory observed
#   2  warmmag T=100   the practical regime — 20% of training to pick the subnetwork
#   3  random  seed42  the control; A.6.1 puts chance Jaccard at ~1.27e-2 at rho=97.5%
#
# Read arm 2 against arm 0: if warm-start at T=100 tracks the oracle, the oracle is an
# instrument rather than a requirement, which is the position the rebuttal takes anyway
# (item 4c, the LT-SFT disclosure).
#
# No checkpoints are written (--save_model false, save_steps past the horizon). The
# deliverable is the reward/loss curve in wandb; 4 x 34 GB of GRPO checkpoints we would
# not read is not worth the scratch.
#SBATCH --job-name=grpo_tsweep_sparse
#SBATCH --output=logs/grpo_tsweep_sparse_%A_%a.out
#SBATCH --error=logs/grpo_tsweep_sparse_%A_%a.err
# d4052-d4055 (the only h200 nodes) sit in BOTH `gpu` and `multigpu`, and the two
# partitions carry SEPARATE QOS counters (each: 8 submitted / 4 running). Listing both
# gives these arms two queues to win from on the same physical hardware instead of one.
# `gpu` caps at 8h, which is already this script's walltime, so nothing is given up.
#SBATCH --partition=gpu,multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-3

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
export WANDB_PROJECT="rl_casino_rebuttal"

STEPS="${STEPS:-500}"
CAP="${CAP:-2048}"
SPARSITY="${SPARSITY:-97.5}"
WARMUP_STEPS=$(python -c "print(int(0.1*$STEPS))" 2>/dev/null || echo 50)

M="$SCRATCH_ROOT/rebuttal_analysis/grpo_tsweep/masks"
OLD="$SCRATCH_ROOT/transfer_v1/oracle_masks_llama8b"
OUT="$SCRATCH_ROOT/rebuttal_analysis/grpo_tsweep/sparse"
mkdir -p "$OUT"

ARM="${SLURM_ARRAY_TASK_ID:-0}"
case "$ARM" in
  0) MASK="$M/grpo_math220k_oracle_T500_sp${SPARSITY}.pt";  TAG="oracle_T500"  ;;
  1) MASK="$M/grpo_math220k_warmmag_T250_sp${SPARSITY}.pt"; TAG="warmmag_T250" ;;
  2) MASK="$M/grpo_math220k_warmmag_T100_sp${SPARSITY}.pt"; TAG="warmmag_T100" ;;
  3) MASK="$OLD/random_baseline_lightr1_sp${SPARSITY}_seed42.pt"; TAG="random_seed42" ;;
  *) echo "FATAL: unknown arm $ARM"; exit 1 ;;
esac

echo "=========================================="
echo "Sparse GRPO T-sweep — arm $ARM ($TAG)"
echo "  steps=$STEPS  scheduler=cosine  warmup=${WARMUP_STEPS}  cap=$CAP  rho=${SPARSITY}%"
echo "  mask=$MASK"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

test -f "$MASK" || { echo "FATAL: mask missing: $MASK"; exit 1; }

python src/full_training/sparse_grpo_bsr.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --mask "$MASK" \
    --dataset math-220k \
    --n_steps "$STEPS" \
    --lr 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps "$WARMUP_STEPS" \
    --grpo_beta 0.025 \
    --max_grad_norm 0.1 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_prompt_length 512 \
    --max_completion_length "$CAP" \
    --grpo_reward_profile llama_cot \
    --optimizer sparse_adamw \
    --save_steps 100000 \
    --save_model false \
    --output_base_dir "$OUT/$TAG" \
    --use_wandb \
    --run_name "grpo_tsweep_${TAG}_${STEPS}steps_cap${CAP}"
RC=$?

echo "=========================================="
echo "arm $ARM ($TAG) exit=$RC  finished $(date)"
echo "=========================================="
exit $RC

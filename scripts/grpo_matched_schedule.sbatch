#!/bin/bash
# Single-variable test of the Appendix D.4 / Figure 9 confound.
#
# THE PROBLEM, read out of training_args.bin in the three checkpoints Figure 9 plots:
#
#            scheduler   max_steps   LR at step 500   % of peak
#   dense    COSINE      5000        4.990e-06        99.8%
#   sparse   LINEAR      500         1.111e-08         0.2%
#
# A 449x gap, with the arms' learning rates moving in OPPOSITE directions across the
# comparison window: dense was still inside its 500-step warmup ramp (warmup_ratio 0.1
# of max_steps 5000), while the sparse arms warmed up in 50 steps and annealed to zero.
# A dense curve that keeps climbing while the sparse curves flatten is the expected
# consequence of that, and it is exactly what Figure 9 shows.
#
# ROOT CAUSE: sparse_grpo_bsr.py exposed no scheduler argument at all, so it silently
# took the HF default (LINEAR) while GRPO_train.py explicitly defaults to COSINE. The
# arms could not have been matched from the command line even deliberately. Fixed on
# this branch by adding --lr_scheduler_type.
#
# THE TEST: rerun both arms with an IDENTICAL schedule and change nothing else.
# max_completion_length stays at the as-run 1024 — raising it as well would make this a
# two-variable test and the diagnosis unreadable. 150 steps, because both outcomes are
# writable:
#     sparse tracks dense  -> D.4's gap was the schedule; the GRPO claim is withdrawn
#     sparse still lags    -> D.4's conclusion may survive; report both, honestly
#
# NOT measured here: wall-clock or throughput. Those stay on the H200 configuration
# used for Tables 2/3 or they are not comparable.
#SBATCH --job-name=grpo_matched
#SBATCH --output=logs/grpo_matched_%A_%a.out
#SBATCH --error=logs/grpo_matched_%A_%a.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-1

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO"; mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_USER_ROOT:-/scratch/$USER}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"
export WANDB_PROJECT="rl_casino_rebuttal"

STEPS="${STEPS:-150}"
CAP="${CAP:-1024}"                # as-run value; deliberately unchanged
WARMUP_STEPS=$(python -c "print(int(0.1*$STEPS))" 2>/dev/null || echo 15)
MASK="$SCRATCH_ROOT/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt"
OUT="$SCRATCH_ROOT/rebuttal_analysis/grpo_matched"
mkdir -p "$OUT"

ARM="${SLURM_ARRAY_TASK_ID:-0}"

echo "=========================================="
echo "GRPO matched-schedule test — arm $ARM"
echo "  steps=$STEPS  scheduler=cosine  warmup=${WARMUP_STEPS} steps  cap=$CAP"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

if [ "$ARM" -eq 0 ]; then
    echo ">>> DENSE"
    python src/full_training/GRPO_train.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset math-220k \
        --num_steps "$STEPS" \
        --learning_rate 5e-6 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --beta 0.025 \
        --max_grad_norm 0.1 \
        --num_generations 8 \
        --generation_batch_size 8 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --max_prompt_length 512 \
        --max_completion_length "$CAP" \
        --save_steps 100000 \
        --output_base_dir "$OUT/dense" \
        --use_wandb \
        --run_name "matched_dense_${STEPS}steps_cap${CAP}"
    RC=$?
else
    echo ">>> SPARSE oracle @97.5%"
    test -f "$MASK" || { echo "FATAL: mask missing: $MASK"; exit 1; }
    # --lr_scheduler_type is NEW here; without it this script silently used linear.
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
        --optimizer sparse_adamw \
        --save_steps 100000 \
        --save_model false \
        --output_base_dir "$OUT/sparse" \
        --use_wandb \
        --run_name "matched_sparse_oracle_${STEPS}steps_cap${CAP}"
    RC=$?
fi

if [ "$RC" -ne 0 ]; then echo "FAILED arm $ARM rc=$RC"; exit "$RC"; fi
echo "Finished arm $ARM: $(date)"

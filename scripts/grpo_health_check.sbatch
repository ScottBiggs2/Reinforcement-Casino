#!/bin/bash
# 20-step GRPO health check: is the completion cap the binding constraint?
#
# WHY, BEFORE ANY 500-STEP GRPO RUN
# All three GRPO arms behind Appendix D.4 ran at max_completion_length=1024 (read from
# training_args.bin, not from logs), while Table 6 of the submission states 2048. At
# 1024, measured over the dense run's first 20 steps:
#
#     completions/clipped_ratio          0.925   <- 92.5% never finish
#     completions/mean_terminated_length 244
#     rewards/accuracy_reward/mean       0.053   <- ~0.4 of 8 rollouts correct
#     accuracy_std / reward_std          0.274   <- accuracy is 27% of the LEARNING signal
#     frac_reward_zero_std               0.025
#
# (Read the last one carefully: GRPO learns from WITHIN-GROUP reward variance, so a
# reward component that is constant across the 8 rollouts contributes exactly zero
# gradient regardless of its magnitude. The format rewards dominate the reward LEVEL
# but not necessarily the signal. accuracy_std/reward_std is the honest share.)
#
# This job reruns the dense arm for 20 steps at 2048 and 3072, changing nothing else,
# and compares against those five numbers.
#
#   If clipped_ratio drops a lot and accuracy share rises -> truncation was the
#   constraint; raise the cap for the real runs and note the step-time cost.
#   If clipped_ratio stays ~0.9 at 3072 -> the model rambles rather than being cut off,
#   the cap is not the lever, and a longer cap only buys step time. Say so and stop.
#
# 20 steps is enough because the effect being tested is large; it is NOT enough to say
# anything about reward trajectories, and no such claim should be drawn from it.
#SBATCH --job-name=grpo_health
#SBATCH --output=logs/grpo_health_%A_%a.out
#SBATCH --error=logs/grpo_health_%A_%a.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
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
export WANDB_MODE=disabled WANDB_DISABLED=true

CAPS=(2048 3072)
CAP="${CAPS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$SCRATCH_ROOT/rebuttal_analysis/grpo_health_cap${CAP}"
mkdir -p "$OUT"

echo "=========================================="
echo "GRPO health check — dense, 20 steps, max_completion_length=$CAP"
echo "Baseline to beat (cap=1024, first 20 steps of the as-run dense job):"
echo "  clipped_ratio 0.925 | term_len 244 | acc_mean 0.053 | acc_signal 0.274"
echo "Job ${SLURM_JOB_ID:-local}  $(date)"
echo "=========================================="

# Locked to the dense GRPO hyperparameters (Table 6 / open_r1_llama31.yaml), with
# max_steps set to the ACTUAL run length. The submitted dense arm used --num_steps 5000
# with warmup_ratio 0.1, so its warmup alone was 500 steps and it was still ramping up
# across the entire window Figure 9 compares. Matching max_steps to the real horizon is
# part of the fix, not a deviation.
python src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --num_steps 20 \
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
    --save_steps 1000 \
    --output_base_dir "$OUT" \
    --run_name "grpo_health_cap${CAP}_20steps"
RC=$?

if [ "$RC" -ne 0 ]; then echo "FAILED rc=$RC"; exit "$RC"; fi

echo ""
echo "=========================================="
echo "Finished cap=$CAP: $(date)"
echo "Compare the logged clipped_ratio / mean_terminated_length /"
echo "accuracy_reward mean+std against the 1024 baseline printed above."
echo "=========================================="

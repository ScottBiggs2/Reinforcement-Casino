#!/bin/bash
# Dense GRPO trajectory that KEEPS its early checkpoints, so warm-start magnitude
# masks can be built at T = 50 / 100 / 250 and compared against the oracle (T = 500).
#
# WHY THIS RUN EXISTS AT ALL
# --------------------------
# The GRPO oracle masks we already hold
#   /scratch/$USER/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step{500,850}_sp97.5.pt
# were built from `transfer_v1/dense_grpo_math220k_llama8b/llama8b_math220k_dense_scott_full`.
# That run has NO checkpoint below step 500. Two reasons, both verified:
#   1. GRPO_train.py defaults `--save_total_limit 3`, so a rolling window of three
#      checkpoints was kept and everything earlier was deleted as training advanced.
#   2. Its own run_manifest.json records `resume_from_checkpoint=checkpoint-300`,
#      i.e. steps 50-300 did exist once and are gone.
# There is therefore no way to build a T=50/100/250 magnitude mask from existing data.
#
# WHY A FRESH TRAJECTORY RATHER THAN REPLICATING THE OLD ONE
# ----------------------------------------------------------
# The old dense run was configured `lr_scheduler_type=cosine, max_steps=5000,
# warmup_ratio=0.1` -> a 500-step warmup. T = 50/100/250 would all land *inside*
# the LR ramp at 10%/20%/50% of peak, so a "the mask stabilises by T" reading would
# be confounded by the LR still ramping. That is the same defect documented in
# docs/run_logs/grpo_schedule_confound_2026-07-25.md.
#
# This run uses the schedule-clean configuration from scripts/grpo_matched_schedule.sbatch
# (cosine over max_steps=500, warmup_ratio 0.1 -> 50 steps) and runs the full 500 steps,
# so a single self-consistent trajectory yields BOTH the oracle (T=500) and the
# warm-start points. Everything else is byte-identical to that script's arm 0.
#
# max_completion_length = 2048, which is Scott's locked spec
# (docs/hyperparams/grpo_500step_5way_sweep.yaml, updated 2026-05-01) and what the
# submission's Table 6 claims. It is NOT the as-run value: every arm in Fig. 9 actually
# ran at 1024, read out of training_args.bin. We take the specified value rather than
# the as-run one because at 1024 `completions/clipped_ratio` sits at 0.5-1.0 and
# ~90% of the optimised reward is formatting, so a T-sweep there would be measuring
# mask quality against a target that barely scores reasoning.
#
# COST: measured 44 min for 150 steps at cap 1024 (job 8734159_0) => ~17.6 s/step.
#       At 2048 generation roughly doubles the tail; budget ~4.5-5 h for 500 steps
#       plus 10 checkpoint writes. 34 GB per checkpoint, ~340 GB total.
#SBATCH --job-name=grpo_tsweep_dense
#SBATCH --output=logs/grpo_tsweep_dense_%j.out
#SBATCH --error=logs/grpo_tsweep_dense_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --requeue

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
OUT="$SCRATCH_ROOT/rebuttal_analysis/grpo_tsweep/dense"
mkdir -p "$OUT"

echo "=========================================="
echo "Dense GRPO T-sweep trajectory"
echo "  steps=$STEPS  scheduler=cosine  warmup_ratio=0.1  cap=$CAP"
echo "  save_steps=50  save_total_limit=20  (default 3 is what destroyed the old run)"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

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
    --grpo_reward_profile llama_cot \
    --save_steps 50 \
    --save_total_limit 20 \
    --resume_from_checkpoint auto \
    --output_base_dir "$OUT" \
    --run_slug grpo_tsweep_dense_${STEPS}steps \
    --use_wandb \
    --run_name "grpo_tsweep_dense_${STEPS}steps_cap${CAP}"
RC=$?

echo "=========================================="
echo "exit=$RC  finished $(date)"
ls -d "$OUT"/grpo_tsweep_dense_${STEPS}steps/checkpoints/checkpoint-* 2>/dev/null | sort -V
echo "=========================================="
exit $RC

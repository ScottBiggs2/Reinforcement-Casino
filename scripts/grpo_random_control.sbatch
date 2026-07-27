#!/bin/bash
# The missing random-mask control for the three sparse GRPO arms quoted in the rebuttal.
#
# WHY THIS EXISTS
# ---------------
# Three sparse GRPO runs already exist and are cited in the reply to bf6h / XD33 / the AC:
#   transfer_v1/sparse_grpo_math220k_oracle_grpo_math          (in-task GRPO mask)
#   transfer_v1/sparse_grpo_math220k_oracle_dpo_lightr1_500    (DPO Light-R1 mask)
#   transfer_v1/sparse_grpo_math220k_oracle_dpo_tulu3          (DPO Tulu3 mask)
# They are single-variable against each other and all three learn (accuracy reward
# 0.0434->0.0833, 0.0587->0.0809, 0.0561->0.0956). What they cannot do is rank masks,
# because there has never been a random-mask GRPO run anywhere on scratch. Without it
# "sparse GRPO trains" is supportable and "the mask matters for GRPO" is not.
#
# WHY NOT JUST WAIT FOR scripts/grpo_tsweep_sparse.sbatch ARM 3
# -------------------------------------------------------------
# That array also has a random seed42 arm, but it is schedule-matched to the NEW dense
# trajectory: cosine, 50-step warmup, completion cap 2048. The three arms above ran
# LINEAR with cap 1024 (read out of training_args.bin, not from the launcher). A cosine
# cap-2048 random arm is the correct control for the T-sweep and the WRONG control for
# the numbers already written into the rebuttal -- comparing across it would reintroduce
# exactly the schedule confound that sank Appendix D.4.
# So: this job replicates the OLD configuration exactly and changes only the mask.
# The T-sweep keeps its own internally-matched random arm. Two controls, two purposes.
#
# CONFIGURATION — copied from training_args.bin of checkpoint-500 of the in-task arm,
# not from any launcher script (run_manifest.json does not record scheduler or caps;
# that omission is what produced three wrong fields in Table 6).
#   lr_scheduler_type LINEAR      max_steps 500        warmup_steps 50 (warmup_ratio 0)
#   learning_rate 5e-6            grpo_beta 0.025      seed 42
#   max_completion_length 1024    max_prompt_length 512
#   per_device_train_batch_size 2 grad_accum 4         num_generations 8
#
# Runtime reference: the in-task arm (job 6372435) took 03:08:32 on one H200, so the 8h
# cap is comfortable and --requeue is defensive only.
#
# READ-OUT: metric is mean rewards/accuracy_reward/mean over steps 0-50 vs 450-500, taken
# from checkpoints/checkpoint-500/trainer_state.json -- identical to how the other three
# arms were read. save_steps/save_total_limit are left at the script defaults (50 / 3) so
# that file exists; --save_model false skips the 16 GB final dump we do not need.

#SBATCH --job-name=grpo_random_control
#SBATCH --output=logs/grpo_random_control_%j.out
#SBATCH --error=logs/grpo_random_control_%j.err
#SBATCH --partition=gpu,multigpu
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

STEPS=500
CAP=1024
SPARSITY=97.5
MASK="$SCRATCH_ROOT/transfer_v1/oracle_masks_llama8b/random_baseline_lightr1_sp${SPARSITY}_seed42.pt"
OUT="$SCRATCH_ROOT/rebuttal_analysis/grpo_random_control"
mkdir -p "$OUT"

echo "=========================================="
echo "Sparse GRPO — RANDOM MASK CONTROL for the 2026-04 arms"
echo "  steps=$STEPS  scheduler=LINEAR  warmup=50  cap=$CAP  rho=${SPARSITY}%"
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
    --lr_scheduler_type linear \
    --warmup_steps 50 \
    --grpo_beta 0.025 \
    --max_grad_norm 1.0 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_prompt_length 512 \
    --max_completion_length "$CAP" \
    --grpo_reward_profile llama_cot \
    --optimizer sparse_adamw \
    --save_model false \
    --output_base_dir "$OUT" \
    --use_wandb \
    --run_name "grpo_random_control_${STEPS}steps_cap${CAP}_linear"
RC=$?

echo "=========================================="
echo "random control exit=$RC  finished $(date)"
echo "=========================================="
exit $RC

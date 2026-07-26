#!/bin/bash
# LoRA PEFT baseline for NeurIPS #29841 rebuttal (AC item 2 / reviewer bf6h W3).
#
# Answers: "the abstract motivates this as an alternative to low-rank PEFT but there
# is no PEFT baseline." Everything except the parameterisation is held identical to
# the dense/sparse DPO runs (Table 7 of the submission): Llama-3.1-8B-Instruct,
# light-r1, 500 steps, eff. batch 128 (bs 2 x ga 64), beta 0.1, warmup 0.1,
# max_len 1024/1024, bf16, adamw_8bit, 1x H200.
#
# LEARNING RATE GRID — deliberately centred low. 1e-4 is the QLoRA *SFT* convention;
# published DPO+LoRA recipes sit an order of magnitude below it (HF alignment-handbook
# QLoRA-DPO ~5e-6, Zephyr full DPO 5e-7). A grid with no point under 5e-5 cannot find
# the optimum and cannot detect the failure mode it is most likely to hit: at high LR,
# DPO drives chosen AND rejected log-probs down together while the margin still grows
# (likelihood displacement, Razin et al. ICLR 2025). Since the margin is our headline
# metric, a blown-up arm posts the best number in the table and generates garbage.
# efficiency_summary.json therefore records logps/chosen and logps/rejected separately.
#
# Array arms (submit as SEPARATE jobs, see NOTE below):
#   0  r=64, lr 5e-6   -> ~168M trainable (2.09%), matched-budget arm vs 97.5% sparse (2.5%)
#   1  r=64, lr 3e-5   -> same budget, mid LR
#   2  r=64, lr 1e-4   -> same budget, high LR (the SFT-convention point)
#   3  r=16, lr 3e-5   -> ~42M (0.52%), the rank axis
#
# The LR sweep is on r=64 because that is the arm the comparison rests on: a mistuned
# r=64 invalidates the matched-budget claim, whereas r=16 changes nobody's mind.
#
# CAVEAT to state in the rebuttal: LoRA gets a tuned LR; dense and sparse both use the
# submitted 5e-7 unchanged, per the pruning convention of retaining the dense schedule
# (Frankle & Carbin ICLR 2019; Renda et al. ICLR 2020). That asymmetry favours LoRA.
#
# SCALING: alpha=2r, so alpha/r=2 is constant across ranks. rsLoRA (Kalajdzievski 2023)
# argues for alpha/sqrt(r), and constant alpha/r under-scales HIGH ranks — i.e. r=64.
# We report the standard alpha=2r; if arm 3 (r=16) beats arms 0-2, rerun r=64 with
# --lora_use_rslora before concluding anything about rank.
#SBATCH --job-name=lora_baseline_8b
#SBATCH --output=logs/lora_baseline_8b_%A_%a.out
#SBATCH --error=logs/lora_baseline_8b_%A_%a.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-3

set -uo pipefail   # NOT -e: we need to inspect the training return code ourselves

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO"
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_USER_ROOT:-/scratch/$USER}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"

RANKS=(64 64 64 16)
LRS=(5e-6 3e-5 1e-4 3e-5)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
R="${RANKS[$IDX]}"
LR="${LRS[$IDX]}"
TAG="arm${IDX}_r${R}_lr${LR}"

# PER-ARM output root. Critical: DPO_train.py writes wandb_run_id.txt and
# run_manifest.json to output_base_dir (NOT the per-run checkpoint dir), so a shared
# root makes concurrent arms race on the wandb run id — after the first requeue two
# arms resume into the same wandb run and clobber each other's history.
OUT_BASE="$SCRATCH_ROOT/rebuttal_lora/$TAG"
mkdir -p "$OUT_BASE"

echo "=========================================="
echo "LoRA baseline $TAG  |  500 steps  |  light-r1"
echo "Job ID : ${SLURM_JOB_ID:-local}   Started: $(date)"
echo "Out    : $OUT_BASE"
echo "=========================================="

# --requeue does NOT fire on TIME LIMIT: an 8h wall hit is CANCELLED, not requeued
# (documented in docs/run_logs/qwen3_32b_autopipeline_2026-06-12.md — this is why the
# 32B chain never advanced). So exit ~40 min early under `timeout` and resubmit from
# the tail, with --resume_from_checkpoint auto picking up the last checkpoint.
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-26400}"   # 7h20m of the 8h wall

timeout --signal=TERM "${TRAIN_TIMEOUT}s" \
python src/full_training/DPO_train.py \
    --model_name "$MODEL_BASE" \
    --dataset light-r1 \
    --lora \
    --lora_r "$R" \
    --learning_rate "$LR" \
    --num_steps 500 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --dpo_beta 0.1 \
    --optim adamw_8bit \
    --gradient_checkpointing \
    --save_steps 100 \
    --save_total_limit 3 \
    --resume_from_checkpoint auto \
    --output_base_dir "$OUT_BASE" \
    --use_wandb \
    --run_name "rebuttal_${TAG}_lightr1_500steps"
TRAIN_RC=$?

echo "training exited rc=$TRAIN_RC (124 = hit TRAIN_TIMEOUT mid-run, expected on a long arm)"

if [ "$TRAIN_RC" -eq 124 ]; then
    echo "Timed out before the wall — resubmitting this arm to continue from its checkpoint."
    sbatch --array="${IDX}" "$REPO/scripts/lora_baseline_8b.sbatch"
elif [ "$TRAIN_RC" -ne 0 ]; then
    echo "FAILED rc=$TRAIN_RC — NOT resubmitting (a real error would just fail again)."
    exit "$TRAIN_RC"
else
    echo "Arm $TAG complete."
    cat "$OUT_BASE/checkpoints/"*/efficiency_summary.json 2>/dev/null || true
fi

echo "=========================================="
echo "Finished $TAG: $(date)"
echo "=========================================="

#!/bin/bash
# Held-out preference eval for the SECOND LoRA learning-rate point (arm2, r=64 @ 1e-4).
#
# WHY: arm0 (r=64 @ 5e-6) gives held-out margin 0.0883 [+0.055,+0.123], acc 0.618
# (job 8763502). With only that one point, the rebuttal cannot say LoRA's rate was
# "tuned" -- one run is a choice, not a sweep, and the claim was softened on 2026-07-27
# for exactly that reason (see DO_NOT_REPEAT entry 9). arm2 finished on 2026-07-27
# (job 8727274_2, 6h25m, 46.07 s/step) with a TRAIN margin of 17.82 against arm0's 10.67,
# but train margins at ~21 epochs are an overfitting readout, not evidence. This job
# supplies the held-out number that decides it.
#
# Every evaluation setting below is copied from the arm0 run so the two are comparable:
# tulu3 (cross-dataset by construction, since both trained on light-r1), holdout tail
# 2000, 500 pairs, beta 0.1, max_length 1024, max_prompt_length 1024.

#SBATCH --job-name=pref_eval_lora_arm2
#SBATCH --output=logs/pref_eval_lora_arm2_%j.out
#SBATCH --error=logs/pref_eval_lora_arm2_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=d1025
#
# HARDWARE IS PINNED ON PURPOSE. The first attempt (8786294) asked for a bare
# `--gres=gpu:1` and Slurm handed it c2207 = v100-pcie, which has no bf16 support and
# sits on PCIe. It got through pass 1 and 104/500 of pass 2 before hitting a 30 min
# wall -- roughly 10x slower than arm0's eval, which ran on d1028 = a100 in 3m12s.
# Two 8B forward passes over 500 pairs is not "small enough for any GPU".

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

LLAMA="meta-llama/Llama-3.1-8B-Instruct"
ADAPTER="$SCRATCH_ROOT/rebuttal_lora/arm2_r64_lr1e-4/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1_lora_r64_lr0p0001/checkpoint-500"
OUT="$SCRATCH_ROOT/rebuttal_analysis/pref_eval"
TAG="lora_r64_lr1e-4"
mkdir -p "$OUT"

test -d "$ADAPTER" || { echo "FATAL: adapter missing: $ADAPTER"; exit 1; }

echo "=========================================="
echo "Held-out preference eval — $TAG"
echo "  adapter : $ADAPTER"
echo "  compare : arm0 lora_r64_lr5e-6 margin 0.0883, acc 0.618"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

python src/evaluation/preference_eval.py \
    --model "$LLAMA" \
    --adapter "$ADAPTER" \
    --ref_model "$LLAMA" \
    --eval_dataset tulu3 \
    --holdout_tail 2000 \
    --n_eval 500 \
    --trained_on light-r1 \
    --beta 0.1 \
    --batch_size 4 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --tag "$TAG" \
    --out "$OUT/${TAG}.json"
RC=$?

echo "Finished $TAG rc=$RC: $(date)"
exit $RC

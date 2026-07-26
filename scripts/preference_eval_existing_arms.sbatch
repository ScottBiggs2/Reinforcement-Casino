#!/bin/bash
# Held-out preference evaluation over the arms we ALREADY trained. No training.
#
# This is the highest-value job in the rebuttal queue: it costs ~30 min per checkpoint,
# retrains nothing, and it replaces the metric that three reviewers independently said
# cannot separate the conditions (X-Q1, X-Q9, J-W2/W3, B-W6).
#
# WHY THE EVAL SET IS CROSS-DATASET, NOT AN IN-DOMAIN HOLDOUT
# src/utils/data_utils.load_dpo_dataset subsets by deterministic PREFIX
# (select(range(n))), never a shuffle, and the submitted runs passed no --subset_size.
# They therefore consumed the WHOLE split: on Light-R1 that is ~3,060 pairs seen ~21
# times. No in-domain holdout exists for those checkpoints and no flag can invent one.
# So each arm is evaluated on the dataset it did NOT train on, which is unseen by
# construction. preference_eval.py labels this CROSS-DATASET on every line of output;
# we report it with that label rather than calling it a holdout.
#
# Reports, per arm: implicit reward margin beta*(log pi - log pi_ref), preference
# accuracy, both with bootstrap CIs, AND the chosen/rejected rewards separately so
# likelihood displacement (Razin et al., ICLR 2025) is visible rather than hidden
# behind a margin.
#SBATCH --job-name=pref_eval
#SBATCH --output=logs/pref_eval_%A_%a.out
#SBATCH --error=logs/pref_eval_%A_%a.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --array=0-8

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

T="$SCRATCH_ROOT/transfer_v1"
OUT="$SCRATCH_ROOT/rebuttal_analysis/pref_eval"
mkdir -p "$OUT"

LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen3-8B"

# tag | policy path | reference (theta^0) | trained_on | eval_dataset
# Light-R1-trained arms are scored on Tulu3 (never seen); the Tulu3-trained arm is
# scored on Light-R1 (never seen). Arm 0 is the untrained base: its margin must be
# ~0 by construction (policy == reference), which is the built-in sanity check --
# if arm 0 does not come out at 0.000, the harness is wrong and nothing else counts.
# NOTE: the Llama Light-R1 DENSE checkpoint is no longer on /scratch (verified
# 2026-07-25: no dense_dpo_*lightr1*llama* directory exists anywhere under it). That is
# the run behind Figure 1 and the 3.24 margin. Its oracle mask survives, so the sparse
# arms below are unaffected, but there is no Llama Light-R1 dense reference to evaluate.
# The Llama dense arm here is the Tulu3 "long" run instead, scored on Light-R1.
S="$T"
CFG=(
  # arm 0 is the sanity check: policy == reference, so the margin MUST come out ~0.000.
  # If it does not, the harness is wrong and no other row means anything.
  "base_llama|$LLAMA|$LLAMA|none|tulu3"
  "sparse_oracle_lightr1|$S/sparse_dpo_lightr1_oracle_step500/sparse_dpo_lightr1_oracle_step500_500steps/final_model|$LLAMA|light-r1|tulu3"
  "sparse_random_lightr1|$S/sparse_dpo_lightr1_random/sparse_dpo_lightr1_random_500steps/final_model|$LLAMA|light-r1|tulu3"
  "sparse_warmmag200_lightr1|$S/sparse_dpo_lightr1_warm_magnitude_step200/sparse_dpo_lightr1_warm_magnitude_step200_500steps/final_model|$LLAMA|light-r1|tulu3"
  "sparse_xtask_oracle_tulu3_on_lightr1|$S/sparse_dpo_lightr1_oracle_dpo_tulu3/sparse_dpo_lightr1_oracle_dpo_tulu3_500steps/final_model|$LLAMA|light-r1|tulu3"
  "sparse_xobj_oracle_grpo_on_lightr1|$S/sparse_dpo_lightr1_oracle_grpo_math/sparse_dpo_lightr1_oracle_grpo_math_500steps/final_model|$LLAMA|light-r1|tulu3"
  "dense_dpo_tulu3_llama|$T/dense_dpo_tulu3_long_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500|$LLAMA|tulu3|light-r1"
  "base_qwen3|$QWEN|$QWEN|none|tulu3"
  "sparse_oracle_lightr1_qwen3|$S/sparse_dpo_light_r1_qwen3_8b_oracle_step500/sparse_dpo_qwen3_8b_light_r1_oracle_step500_500steps/final_model|$QWEN|light-r1|tulu3"
)

IFS='|' read -r TAG MODEL REF TRAINED EVALDS <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Preference eval: $TAG"
echo "  policy    : $MODEL"
echo "  reference : $REF"
echo "  trained on: $TRAINED   ->  evaluated on: $EVALDS"
echo "  job ${SLURM_JOB_ID:-local}  $(date)"
echo "=========================================="

if [ ! -d "$MODEL" ] && [[ "$MODEL" == /* ]]; then
    echo "SKIP: policy path does not exist: $MODEL"
    exit 0
fi

python src/evaluation/preference_eval.py \
    --model "$MODEL" \
    --ref_model "$REF" \
    --eval_dataset "$EVALDS" \
    --holdout_tail 2000 \
    --n_eval 500 \
    --trained_on "$TRAINED" \
    --beta 0.1 \
    --batch_size 4 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --tag "$TAG" \
    --out "$OUT/${TAG}.json"

echo "Finished $TAG: $(date)"

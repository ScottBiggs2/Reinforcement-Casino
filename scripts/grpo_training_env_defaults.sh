# shellcheck shell=bash
# Canonical GRPO training environment defaults — MUST stay aligned with
# scripts/grpo_openr1_llama31_slurm.sh and docs/hyperparams/open_r1_llama31.yaml
#
# Source from repo-root scripts AFTER RL_CASINO_SCRATCH_ROOT (and thus HF cache roots)
# are set, e.g.:
#   # shellcheck source=/dev/null
#   source "${REPO_ROOT}/scripts/grpo_training_env_defaults.sh"
#
# Purpose: parent Slurm jobs that `sbatch` nested training children with
# `--export=ALL,...` must have these variables **exported** so sparse runs match
# a plain `sbatch scripts/grpo_openr1_llama31_slurm.sh` dense run.

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"

export GRPO_MODE="${GRPO_MODE:-dense}"
export GRPO_NGPUS="${GRPO_NGPUS:-1}"

export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-1000}"
export GRPO_RESUME="${GRPO_RESUME:-}"

export GRPO_LR="${GRPO_LR:-5e-6}"
export GRPO_BETA="${GRPO_BETA:-0.025}"
export GRPO_PER_DEVICE_BS="${GRPO_PER_DEVICE_BS:-2}"
export GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-4}"
export GRPO_NUM_GEN="${GRPO_NUM_GEN:-8}"
export GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-8}"

export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"
export GRPO_OPTIM="${GRPO_OPTIM:-adamw_8bit}"
export GRPO_PRECISION="${GRPO_PRECISION:-bf16}"

# Sequence caps — YAML + grpo_openr1_llama31_slurm.sh use 512 / 1024 (not 2048).
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-1024}"

export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"

# Match dense GRPO_train.py: warmup_ratio=0.1, max_grad_norm=0.1 (sparse gets integer warmup_steps).
export GRPO_WARMUP_RATIO="${GRPO_WARMUP_RATIO:-0.1}"
export GRPO_MAX_GRAD_NORM="${GRPO_MAX_GRAD_NORM:-0.1}"

export GRPO_SPARSE_ADAMW_LAZY="${GRPO_SPARSE_ADAMW_LAZY:-0}"

export GRPO_RUN_SLUG="${GRPO_RUN_SLUG:-}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-}"

export GRPO_DELTA_LOG_INTERVAL="${GRPO_DELTA_LOG_INTERVAL:-}"
export GRPO_DELTA_LOG_END_STEP="${GRPO_DELTA_LOG_END_STEP:-}"

# Optional: match login-node / driver expectations for training jobs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

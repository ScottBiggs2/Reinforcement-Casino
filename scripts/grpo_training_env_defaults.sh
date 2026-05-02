# shellcheck shell=bash
# Canonical GRPO training environment — MUST match docs/hyperparams/open_r1_llama31.yaml
#
# DEFAULT BEHAVIOR (production): **frozen** training hyperparameters. `sbatch --export=ALL`
# and login-shell exports cannot change LR, steps, sequence caps, batch sizes, etc. This is
# what went wrong before: inherited env vars silently overrode “defaults”.
#
# Ablations / long runs: export GRPO_HPARAM_OVERRIDE=1 before sbatch, then set any GRPO_*
# variables you need. The launcher will print "GRPO_HPARAM_MODE=override".
#
# Source from repo root after RL_CASINO_SCRATCH_ROOT is set:
#   # shellcheck source=/dev/null
#   source "${REPO_ROOT}/scripts/grpo_training_env_defaults.sh"

# ---------------------------------------------------------------------------
# Frozen Open-R1 GRPO (Llama 3.1 8B) — single source; edit only here + YAML
# ---------------------------------------------------------------------------
_grpo_freeze_training_hparams() {
  export GRPO_TARGET_STEPS=500
  export GRPO_LR=5e-6
  export GRPO_BETA=0.025
  export GRPO_PER_DEVICE_BS=2
  export GRPO_GRAD_ACCUM=4
  export GRPO_NUM_GEN=8
  export GRPO_GEN_BATCH=8
  export GRPO_SAVE_STEPS=50
  export GRPO_SAVE_TOTAL_LIMIT=3
  export GRPO_OPTIM=adamw_8bit
  export GRPO_PRECISION=bf16
  export GRPO_MAX_PROMPT_LENGTH=512
  export GRPO_MAX_COMPLETION_LENGTH=2048
  export GRPO_REWARD_PROFILE=llama_cot
  export GRPO_WARMUP_RATIO=0.1
  export GRPO_MAX_GRAD_NORM=0.1
  export GRPO_SPARSE_ADAMW_LAZY=0
  # Model + dataset: fixed for the standard paper / Open-R1 Math pipeline
  export MODEL="meta-llama/Llama-3.1-8B-Instruct"
  export GRPO_DATASET="math-220k"
}

# ---------------------------------------------------------------------------
# Path + runtime (still env-configurable; not training hyperparameters)
# ---------------------------------------------------------------------------
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"

export GRPO_MODE="${GRPO_MODE:-dense}"
export GRPO_NGPUS="${GRPO_NGPUS:-1}"
export GRPO_RESUME="${GRPO_RESUME:-}"

export GRPO_RUN_SLUG="${GRPO_RUN_SLUG:-}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-}"

export GRPO_DELTA_LOG_INTERVAL="${GRPO_DELTA_LOG_INTERVAL:-}"
export GRPO_DELTA_LOG_END_STEP="${GRPO_DELTA_LOG_END_STEP:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

# ---------------------------------------------------------------------------
# Training hyperparameters: frozen vs override
# ---------------------------------------------------------------------------
if [ "${GRPO_HPARAM_OVERRIDE:-0}" = "1" ]; then
  export GRPO_HPARAM_MODE="override"
  export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-500}"
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
  export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
  export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-2048}"
  export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
  export GRPO_WARMUP_RATIO="${GRPO_WARMUP_RATIO:-0.1}"
  export GRPO_MAX_GRAD_NORM="${GRPO_MAX_GRAD_NORM:-0.1}"
  export GRPO_SPARSE_ADAMW_LAZY="${GRPO_SPARSE_ADAMW_LAZY:-0}"
  export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
else
  export GRPO_HPARAM_MODE="locked"
  _grpo_freeze_training_hparams
fi

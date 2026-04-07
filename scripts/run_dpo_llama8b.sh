#!/bin/bash
# Run Dense DPO training with Llama-3.1-8B-Instruct on multi-GPU.
# Purpose: Generate weight deltas for warm-start mask generation.
# Usage:
#   sbatch scripts/run_dpo_llama8b.sh
#   RUN_TAG=light_r1 sbatch scripts/run_dpo_llama8b.sh
#   sbatch --gres=gpu:v100-sxm2:4 scripts/run_dpo_llama8b.sh  # override GPU type
#SBATCH --job-name=llama8b_dpo
#SBATCH --output=logs/llama8b_dpo_%j.out
#SBATCH --error=logs/llama8b_dpo_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=08:00:00

# === CONFIGURATION ===
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="light-r1"
N_STEPS=500
BATCH_SIZE=1
LR=1e-6
DPO_BETA=0.1
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
DELTA_LOG_INTERVAL=50
WANDB_PROJECT="huggingface"

# === STORAGE (all on /scratch) ===
export HF_HOME="/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/$USER/hf_cache/datasets"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
COMMON_OUTPUT_DIR="/scratch/$USER/rl_casino_outputs"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$COMMON_OUTPUT_DIR"

# Copy HF token to scratch HF_HOME if it exists in default location
if [ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

# === ENVIRONMENT SETUP ===
source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/$USER/.conda/envs/rl_casino || conda activate /scratch/$USER/conda_envs/rl_casino

export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=$WANDB_PROJECT

# === REPO ROOT ===
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# === MULTI-GPU (auto-detect from SLURM) ===
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NGPUS="$SLURM_GPUS_ON_NODE"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NGPUS=1
fi
GRAD_ACCUM=$((8 / NGPUS))
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

if [ "$NGPUS" -gt 1 ]; then
    LAUNCH="accelerate launch --num_processes $NGPUS --multi_gpu"
else
    LAUNCH="python"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="llama8b_dpo_${RUN_TAG:-dense}_${NGPUS}gpu_${TIMESTAMP}"

echo "=================================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPUs: $NGPUS | Grad accum: $GRAD_ACCUM"
echo "HF_HOME: $HF_HOME"
echo "Output dir: $COMMON_OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo "=================================================="

$LAUNCH src/full_training/DPO_train.py \
    --model_name "$MODEL" \
    --dataset "$DATASET" \
    --num_steps "$N_STEPS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --dpo_beta "$DPO_BETA" \
    --max_length "$MAX_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --delta_log_interval "$DELTA_LOG_INTERVAL" \
    --optim adamw_8bit \
    --gradient_checkpointing \
    --output_base_dir "$COMMON_OUTPUT_DIR" \
    --dataset_cache_dir "$HF_DATASETS_CACHE" \
    --use_wandb \
    --run_name "$RUN_NAME" 2>&1

STATUS=$?
echo ""
echo "=================================================="
echo "Job finished at: $(date)"
if [ $STATUS -eq 0 ]; then
    echo "SUCCESS: ${RUN_NAME}"
    echo "Deltas saved to: ${COMMON_OUTPUT_DIR}/deltas/"
else
    echo "FAILED: ${RUN_NAME} (exit code $STATUS)"
fi
echo "=================================================="
exit $STATUS

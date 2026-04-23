#!/bin/bash
# Dense DPO training with Llama-3.1-8B-Instruct — hyperparams matched to
# run_dpo_masks_llama8b.sh so dense vs sparse DPO probes are comparable.
# Uses src/full_training/DPO_train.py (full-weight DPO, no mask).
#
# Usage:
#   RUN_TAG=dense_dpo_matched sbatch scripts/run_dpo_dense_llama8b.sh
#
#SBATCH --job-name=llama8b_dense_dpo
#SBATCH --output=logs/llama8b_dense_dpo_%j.out
#SBATCH --error=logs/llama8b_dense_dpo_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=07:45:00

set -euo pipefail

# === CONFIGURATION === (env-overridable for monitor auto-intervention)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-tulu3}"
N_STEPS="${N_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
DPO_BETA="${DPO_BETA:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"

export HF_HOME="/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/$USER/hf_cache/datasets"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
COMMON_OUTPUT_DIR="/scratch/$USER/rl_casino_outputs"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$COMMON_OUTPUT_DIR"

if [ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/$USER/.conda/envs/rl_casino || conda activate /scratch/$USER/conda_envs/rl_casino

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export WANDB_PROJECT=$WANDB_PROJECT

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then NGPUS="$SLURM_GPUS_ON_NODE";
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l);
else NGPUS=1; fi
GRAD_ACCUM=$((8 / NGPUS)); [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

if [ "$NGPUS" -gt 1 ]; then
    LAUNCH="accelerate launch --num_processes $NGPUS --multi_gpu"
else
    LAUNCH="python"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="llama8b_dense_dpo_${RUN_TAG:-matched}_${NGPUS}gpu_${TIMESTAMP}"

echo "=================================================="
echo "Dense DPO MATCHED: lr=$LR n_steps=$N_STEPS"
echo "Run name: $RUN_NAME"
echo "=================================================="

# delta_log_end_step set to N_STEPS so deltas save across the full run;
# without this the default caps at ~step 50 and we lose trajectory data
# needed to compute the NEW magnitude mask for sparse-Oracle training.
$LAUNCH src/full_training/DPO_train.py \
    --model_name "$MODEL" \
    --dataset "$DATASET" \
    --num_steps "$N_STEPS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --dpo_beta "$DPO_BETA" \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --delta_log_interval 50 \
    --delta_log_end_step "$N_STEPS" \
    --optim adamw_8bit \
    --gradient_checkpointing \
    --output_base_dir "$COMMON_OUTPUT_DIR" \
    --dataset_cache_dir "$HF_DATASETS_CACHE" \
    --use_wandb \
    --run_name "$RUN_NAME" 2>&1

STATUS=$?
echo "=================================================="
echo "Finished: $(date)"
if [ $STATUS -eq 0 ]; then
    echo "SUCCESS: ${RUN_NAME}"
else
    echo "FAILED: ${RUN_NAME} (exit ${STATUS})"
fi
echo "=================================================="
exit $STATUS

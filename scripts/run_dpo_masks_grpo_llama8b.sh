#!/bin/bash
# Run GRPO training (sparse or dense) with Llama-3.1-8B-Instruct on H200 multi-GPU.
# Usage:
#   MASK_PATH=/path/to/mask.pt RUN_TAG=cav_dpo sbatch scripts/run_dpo_masks_grpo_llama8b.sh
#   RUN_TAG=dense_baseline sbatch scripts/run_dpo_masks_grpo_llama8b.sh   # no mask = dense
#
# Scale GPUs (default 4):
#   sbatch --gres=gpu:h200:2 scripts/run_dpo_masks_grpo_llama8b.sh
#   sbatch --gres=gpu:h200:8 scripts/run_dpo_masks_grpo_llama8b.sh
#   sbatch --gres=gpu:a100:4 scripts/run_dpo_masks_grpo_llama8b.sh       # different GPU type
#SBATCH --job-name=llama8b_grpo
#SBATCH --output=logs/llama8b_grpo_%j.out
#SBATCH --error=logs/llama8b_grpo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=400G
#SBATCH --time=04:00:00

# ── Repo root ────────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Working dir: $(pwd)"

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_ENV_PRIMARY="/scratch/xie.yiyi/conda_envs/rl_casino"
CONDA_ENV_FALLBACK="/home/xie.yiyi/.conda/envs/rl_casino"

if [ -d "$CONDA_ENV_PRIMARY" ]; then
    ENV_PATH="$CONDA_ENV_PRIMARY"
elif [ -d "$CONDA_ENV_FALLBACK" ]; then
    ENV_PATH="$CONDA_ENV_FALLBACK"
else
    echo "ERROR: No conda env found at $CONDA_ENV_PRIMARY or $CONDA_ENV_FALLBACK"
    exit 1
fi

PYTHON_BIN="$ENV_PATH/bin/python"
ACCELERATE_BIN="$ENV_PATH/bin/accelerate"
export PATH="$ENV_PATH/bin:$PATH"
export PYTHONPATH=.
echo "Using env: $ENV_PATH"

echo "Installing/verifying requirements..."
if "$PYTHON_BIN" -c "import trl" 2>/dev/null; then
    echo "Requirements already satisfied."
else
    "$PYTHON_BIN" -m pip install -r requirements.txt -q
fi

# ── Multi-GPU config (auto-detect from SLURM allocation) ────────────────────
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NGPUS="$SLURM_GPUS_ON_NODE"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NGPUS=1
fi
echo "Detected $NGPUS GPUs"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="math-220k"
N_STEPS=200
SUBSET=512
BATCH_SIZE=1
GRAD_ACCUM=$((8 / NGPUS))            # effective batch stays the same across GPU counts
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1
NUM_GENERATIONS=8
GEN_BATCH_SIZE=8
LR=1e-6
OUTPUT_DIR="/home/xie.yiyi/Reinforcement-Casino/mask_swapping"
CACHE_DIR="/scratch/xie.yiyi/hf_cache/datasets"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

RUN_NAME="llama8b_dpo_mask_grpo_${RUN_TAG:-unknown}_${NGPUS}gpu_${TIMESTAMP}"

# ── Launch command ───────────────────────────────────────────────────────────
if [ "$NGPUS" -gt 1 ]; then
    LAUNCH="$ACCELERATE_BIN launch --num_processes $NGPUS --multi_gpu"
else
    LAUNCH="$PYTHON_BIN"
fi

# ── Run ──────────────────────────────────────────────────────────────────────
if [ -n "${MASK_PATH:-}" ] && [ -f "$MASK_PATH" ]; then
    echo "============================================================"
    echo "Sparse GRPO with mask: ${MASK_PATH}"
    echo "Model: ${MODEL}  |  GPUs: ${NGPUS}"
    echo "Run name: ${RUN_NAME}"
    echo "============================================================"

    $LAUNCH src/full_training/sparse_grpo_bsr.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --n_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --lr "$LR" \
        --optimizer sparse_adamw \
        --mask "$MASK_PATH" \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$CACHE_DIR" \
        --use_wandb \
        --save_model true \
        --run_name "$RUN_NAME" 2>&1
else
    echo "============================================================"
    echo "Dense GRPO Baseline (no mask)"
    echo "Model: ${MODEL}  |  GPUs: ${NGPUS}"
    echo "Run name: ${RUN_NAME}"
    echo "============================================================"

    $LAUNCH src/full_training/GRPO_train.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --num_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$CACHE_DIR" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --use_wandb \
        --run_name "$RUN_NAME" 2>&1
fi

STATUS=$?
echo ""
echo "Job finished at: $(date)"
if [ $STATUS -eq 0 ]; then
    echo "✓ ${RUN_NAME}: SUCCESS"
else
    echo "✗ ${RUN_NAME}: FAILED (exit code $STATUS)"
fi
exit $STATUS

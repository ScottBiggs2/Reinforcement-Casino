#!/bin/bash
# Run GRPO training with DPO-generated masks to evaluate cross-method transfer.
# Runs: 1 dense baseline + 1 sparse GRPO per DPO mask found in MASK_DIR.
# Submit: sbatch scripts/run_dpo_masks_grpo.sh
#SBATCH --job-name=dpo_mask_grpo
#SBATCH --output=logs/dpo_mask_grpo_%j.out
#SBATCH --error=logs/dpo_mask_grpo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00

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
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
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
export PATH="$ENV_PATH/bin:$PATH"
export PYTHONPATH=.
echo "Using env: $ENV_PATH"

echo "Installing/verifying requirements..."
if "$PYTHON_BIN" -c "import trl" 2>/dev/null; then
    echo "Requirements already satisfied."
else
    "$PYTHON_BIN" -m pip install -r requirements.txt -q
fi

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="google/gemma-3-270m-it"
MASK_DIR="/home/xie.yiyi/Reinforcement-Casino/masks/grpo_verify"
DATASET="math-220k"
N_STEPS=200
SUBSET=512
BATCH_SIZE=1
GRAD_ACCUM=8
NUM_GENERATIONS=8
GEN_BATCH_SIZE=8
LR=5e-6
OPTIMIZER="sparse_adamw"
OUTPUT_DIR="/home/xie.yiyi/Reinforcement-Casino/mask_swapping"
CACHE_DIR="/scratch/xie.yiyi/hf_cache/datasets"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL=0

# ── Helper ───────────────────────────────────────────────────────────────────
run_sparse_grpo() {
    local MASK_PATH="$1"
    local MASK_NAME="$2"
    local RUN_NAME="dpo_mask_grpo_${MASK_NAME}_${TIMESTAMP}"

    echo ""
    echo "============================================================"
    echo "Sparse GRPO with DPO mask: ${MASK_NAME}"
    echo "  Mask: ${MASK_PATH}"
    echo "  Run:  ${RUN_NAME}"
    echo "============================================================"

    "$PYTHON_BIN" src/full_training/sparse_grpo_bsr.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --n_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --lr "$LR" \
        --optimizer "$OPTIMIZER" \
        --mask "$MASK_PATH" \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$CACHE_DIR" \
        --use_wandb \
        --save_model true \
        --run_name "$RUN_NAME" 2>&1

    TOTAL=$((TOTAL + 1))
    if [ $? -eq 0 ]; then
        echo "  ✓ ${MASK_NAME}: PASSED"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ ${MASK_NAME}: FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ── Run 0: Dense baseline ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Dense GRPO Baseline (no mask)"
echo "============================================================"

"$PYTHON_BIN" src/full_training/GRPO_train.py \
    --model_name "$MODEL" \
    --dataset "$DATASET" \
    --num_steps "$N_STEPS" \
    --subset_size "$SUBSET" \
    --output_base_dir "$OUTPUT_DIR" \
    --dataset_cache_dir "$CACHE_DIR" \
    --num_generations "$NUM_GENERATIONS" \
    --generation_batch_size "$GEN_BATCH_SIZE" \
    --use_wandb \
    --run_name "dpo_mask_grpo_dense_baseline_${TIMESTAMP}" 2>&1

TOTAL=$((TOTAL + 1))
if [ $? -eq 0 ]; then
    echo "  ✓ Dense baseline: PASSED"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "  ✗ Dense baseline: FAILED"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# ── Run 1-N: Sparse GRPO with each DPO mask ─────────────────────────────────
MASK_COUNT=0
for MASK_FILE in "${MASK_DIR}"/*.pt; do
    if [ ! -f "$MASK_FILE" ]; then
        echo "WARNING: No .pt mask files found in ${MASK_DIR}"
        break
    fi
    MASK_COUNT=$((MASK_COUNT + 1))
    # Extract mask name from filename (strip path and extension)
    MASK_BASENAME=$(basename "$MASK_FILE" .pt)
    run_sparse_grpo "$MASK_FILE" "$MASK_BASENAME"
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Masks found:  ${MASK_COUNT}"
echo "Total runs:   ${TOTAL} (1 dense + ${MASK_COUNT} sparse)"
echo "Passed:       ${PASS_COUNT} / ${TOTAL}"
echo "Failed:       ${FAIL_COUNT} / ${TOTAL}"
echo ""
echo "Outputs:      ${OUTPUT_DIR}"
echo "Job finished at: $(date)"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    echo "ALL RUNS COMPLETED SUCCESSFULLY"
    exit 0
fi

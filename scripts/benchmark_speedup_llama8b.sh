#!/bin/bash
# ============================================================================
# Sparse Training Speedup Benchmark — Llama-3.1-8B-Instruct — Single H200 GPU
#
# Runs 4 training methods with identical hyperparams and collects timing JSON:
#   1. Dense + AdamW
#   2. Dense + SGD
#   3. LoRA + AdamW
#   4. Sparse BSR + SparseAdamW
#
# Submit: sbatch scripts/benchmark_speedup_llama8b.sh
# ============================================================================
#SBATCH --job-name=bench_8b
#SBATCH --output=logs/bench_8b_%j.out
#SBATCH --error=logs/bench_8b_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail

# ── Repo root ───────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

echo "====== Llama-3.1-8B Speedup Benchmark ======"
echo "Date:     $(date)"
echo "Node:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-auto}"

# ── Environment ─────────────────────────────────────────────────────────────
source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/$USER/.conda/envs/rl_casino || conda activate /scratch/$USER/conda_envs/rl_casino

export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="/scratch/$USER/pip_packages:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Storage — all on /scratch
export HF_HOME="/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/$USER/hf_cache/datasets"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

# Copy HF token if needed (for gated Llama model)
if [ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
fi

echo "Env:      $(which python)"
echo "HF_HOME:  $HF_HOME"

# ── Shared Config ───────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="math-220k"
N_STEPS=50
SUBSET=512
BATCH_SIZE=1
GRAD_ACCUM=8
NUM_GENERATIONS=8
GEN_BATCH_SIZE=8
LR=1e-6

# Mask for sparse run — update this to match your available mask
MASK_PATH="${MASK_PATH:-masks/grpo_verify/fisher_grpo.pt}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/scratch/$USER/rl_casino_outputs/benchmark_llama8b_${TIMESTAMP}"
CACHE_DIR="$HF_DATASETS_CACHE"
mkdir -p "$OUTPUT_DIR"

PASS=0
FAIL=0
TOTAL=0

run_and_track() {
    local NAME="$1"; shift
    echo ""
    echo "============================================================"
    echo "[$((TOTAL+1))] $NAME"
    echo "============================================================"
    TOTAL=$((TOTAL + 1))
    if "$@"; then
        echo "  -> $NAME: PASSED"
        PASS=$((PASS + 1))
    else
        echo "  -> $NAME: FAILED"
        FAIL=$((FAIL + 1))
    fi
}

# ── 1. Dense + AdamW ────────────────────────────────────────────────────────
run_and_track "Dense + AdamW" \
    python src/full_training/GRPO_timing_baseline.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --n_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --lr "$LR" \
        --optimizer adamw \
        --use_wandb \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$CACHE_DIR"

# ── 2. Dense + SGD ──────────────────────────────────────────────────────────
run_and_track "Dense + SGD" \
    python src/full_training/GRPO_timing_baseline.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --n_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --lr "$LR" \
        --optimizer sgd \
        --use_wandb \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$CACHE_DIR"

# ── 3. LoRA + AdamW ─────────────────────────────────────────────────────────
run_and_track "LoRA + AdamW (r=16)" \
    python src/full_training/lora_grpo_timing.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --n_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --lr "$LR" \
        --optimizer adamw \
        --lora_rank 16 \
        --use_wandb \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$CACHE_DIR"

# ── 4. Sparse BSR + SparseAdamW ─────────────────────────────────────────────
# Requires a mask file matching the 8B model architecture.
# Skip if no mask is available; generate one first with the mask pipeline.
if [ -f "$MASK_PATH" ]; then
    run_and_track "Sparse BSR + SparseAdamW" \
        python src/full_training/sparse_grpo_bsr.py \
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
            --block_size_bsr 16 \
            --block_size_adam 128 \
            --use_wandb \
            --output_base_dir "$OUTPUT_DIR" \
            --dataset_cache_dir "$CACHE_DIR" \
            --save_model false
else
    echo ""
    echo "============================================================"
    echo "SKIPPING Sparse BSR — no mask file found at $MASK_PATH"
    echo "Generate a mask first, then re-run with MASK_PATH=<path>"
    echo "============================================================"
fi

# ── Collect Results ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Collecting timing results..."
echo "============================================================"

python scripts/generate_report.py "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
echo "============================================================"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo "Output: $OUTPUT_DIR"
echo "Finished: $(date)"

[ "$FAIL" -gt 0 ] && exit 1 || exit 0

#!/bin/bash
#SBATCH --job-name=probe_analysis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_analysis_%j.out
#SBATCH --error=logs/probe_analysis_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started  : $(date)"
echo "========================================"

# ── GPU driver ────────────────────────────────────────────────────────
CUDA_LIB=$(dirname "$(find /usr /opt /lib64 /lib -name "libcuda.so.1" 2>/dev/null | head -n1)")
if [ -n "$CUDA_LIB" ] && [ "$CUDA_LIB" != "." ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[gpu] injected libcuda from: $CUDA_LIB"
else
    echo "[warn] libcuda.so.1 not found on this node"
fi

# ── Conda environment ────────────────────────────────────────────────
CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PRIMARY="/scratch/xie.yiyi/conda_envs/rl_casino"
CONDA_ENV_FALLBACK="/home/xie.yiyi/.conda/envs/rl_casino"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV:-}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
fi

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    if [ -d "$CONDA_ENV_PRIMARY" ]; then
        conda activate "$CONDA_ENV_PRIMARY"
        PYTHON_BIN="python"
        echo "[env] Activated conda env: $CONDA_ENV_PRIMARY"
    elif [ -d "$CONDA_ENV_FALLBACK" ]; then
        conda activate "$CONDA_ENV_FALLBACK"
        PYTHON_BIN="python"
        echo "[env] Activated conda env: $CONDA_ENV_FALLBACK"
    else
        echo "[warn] Conda env not found at either path; using $PYTHON_BIN"
    fi
else
    echo "[warn] conda.sh not found at $CONDA_SH; using $PYTHON_BIN"
fi

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs

# ── Configuration ─────────────────────────────────────────────────────
MODEL="${MODEL:-google/gemma-3-270m-it}"
MASK_DIR="${MASK_DIR:-masks/grpo_verify}"
OUTPUT_DIR="${OUTPUT_DIR:-probe_results/fisher_dpo_vs_grpo}"

MASK_A="${MASK_A:-$MASK_DIR/fisher_dpo.pt}"
MASK_B="${MASK_B:-$MASK_DIR/fisher_grpo.pt}"
MASK_A_LABEL="${MASK_A_LABEL:-Fisher-DPO}"
MASK_B_LABEL="${MASK_B_LABEL:-Fisher-GRPO}"

LAYER_STRIDE="${LAYER_STRIDE:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-128}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"

echo "[config] MODEL=$MODEL"
echo "[config] MASK_A=$MASK_A"
echo "[config] MASK_B=$MASK_B"
echo "[config] OUTPUT_DIR=$OUTPUT_DIR"
echo "[config] LAYER_STRIDE=$LAYER_STRIDE"
echo "[config] BATCH_SIZE=$BATCH_SIZE"
echo "[config] INCLUDE_BASELINE=$INCLUDE_BASELINE"

# ── Validate masks exist ─────────────────────────────────────────────
for f in "$MASK_A" "$MASK_B"; do
    if [ ! -f "$f" ]; then
        echo "[error] Mask file not found: $f"
        exit 1
    fi
done

# ── Run probe analysis ───────────────────────────────────────────────
BASELINE_FLAG=""
if [ "$INCLUDE_BASELINE" = "1" ]; then
    BASELINE_FLAG="--include_baseline"
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Running probe analysis ..."
echo "========================================"

$PYTHON_BIN src/analysis/probe_analysis.py \
    --model "$MODEL" \
    --mask_a "$MASK_A" \
    --mask_b "$MASK_B" \
    --mask_a_label "$MASK_A_LABEL" \
    --mask_b_label "$MASK_B_LABEL" \
    --output_dir "$OUTPUT_DIR" \
    --layer_stride "$LAYER_STRIDE" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    $BASELINE_FLAG

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results in: $OUTPUT_DIR/"
echo "  - probe_results.json"
echo "  - probe_heatmap.png"
echo "========================================"

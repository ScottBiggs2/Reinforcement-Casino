#!/bin/bash
#SBATCH --job-name=probe_pair_12masks
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/probe_pair_12masks_%j.out
#SBATCH --error=logs/probe_pair_12masks_%j.err

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
fi

# ── Conda environment ────────────────────────────────────────────────
CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PRIMARY="/scratch/xie.yiyi/conda_envs/rl_casino"
CONDA_ENV_FALLBACK="/home/xie.yiyi/.conda/envs/rl_casino"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    if [ -d "$CONDA_ENV_PRIMARY" ]; then
        conda activate "$CONDA_ENV_PRIMARY"
    elif [ -d "$CONDA_ENV_FALLBACK" ]; then
        conda activate "$CONDA_ENV_FALLBACK"
    fi
    PYTHON_BIN="python"
    echo "[env] Python: $(which python)"
fi

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs

# ── Configuration ─────────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/xie.yiyi/probe_pair_12masks}"
LAYER_STRIDE="${LAYER_STRIDE:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CV_FOLDS="${CV_FOLDS:-5}"
PAIRS_PER_POS="${PAIRS_PER_POS:-2}"

echo "[config] MODEL=$MODEL"
echo "[config] OUTPUT_DIR=$OUTPUT_DIR"
echo "[config] LAYER_STRIDE=$LAYER_STRIDE"
echo "[config] BATCH_SIZE=$BATCH_SIZE"
echo "[config] PAIRS_PER_POS=$PAIRS_PER_POS"

mkdir -p "$OUTPUT_DIR"

$PYTHON_BIN src/analysis/probe_pair_12masks.py \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --layer_stride "$LAYER_STRIDE" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --cv_folds "$CV_FOLDS" \
    --pairs_per_pos "$PAIRS_PER_POS"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR/"
echo "  - probe_pair_results.json"
echo "  - probe_pair_heatmap_all.png"
echo "  - probe_pair_delta_all.png"
echo "========================================"

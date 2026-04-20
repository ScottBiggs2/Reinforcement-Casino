#!/bin/bash
#SBATCH --job-name=verify_dense
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/verify_dense_%j.out
#SBATCH --error=logs/verify_dense_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started  : $(date)"
echo "========================================"

CUDA_LIB=$(dirname "$(find /usr /opt /lib64 /lib -name "libcuda.so.1" 2>/dev/null | head -n1)")
if [ -n "$CUDA_LIB" ] && [ "$CUDA_LIB" != "." ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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
fi

REPO_ROOT="${REPO_ROOT:-/home/xie.yiyi/Reinforcement-Casino}"
cd "$REPO_ROOT"
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/xie.yiyi/probe_verify_dense}"
PROBE_CACHE="${PROBE_CACHE:-/scratch/xie.yiyi/probe_pair_cav_random_oracle_holdout/probe_dataset_cache.json}"

EXTRA=()
if [ -f "$PROBE_CACHE" ]; then
    EXTRA+=(--probe_cache "$PROBE_CACHE")
fi

"$PYTHON_BIN" src/analysis/verify_dense_baseline.py \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --layer_stride "${LAYER_STRIDE:-4}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --max_length "${MAX_LENGTH:-256}" \
    --cv_folds "${CV_FOLDS:-5}" \
    --pairs_per_pos "${PAIRS_PER_POS:-2}" \
    --n_jobs "${N_JOBS:-8}" \
    --holdout_frac "${HOLDOUT_FRAC:-0.2}" \
    --shuffle_seed "${SHUFFLE_SEED:-1337}" \
    "${EXTRA[@]}"

echo ""
echo "Done: $(date)"
echo "Results in: $OUTPUT_DIR/"
echo "  - verify_dense_results.json"
echo "  - verify_dense.png"

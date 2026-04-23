#!/bin/bash
#SBATCH --job-name=mask_erank_cka
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mask_erank_cka_%j.out
#SBATCH --error=logs/mask_erank_cka_%j.err

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
MASKS_JSON="${MASKS_JSON:-scripts/probe_pair_masks_cav_random_oracle.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/xie.yiyi/mask_erank_cka}"

"$PYTHON_BIN" src/analysis/inspect_mask_erank_cka.py \
    --model "$MODEL" \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --layer_substr "${LAYER_SUBSTR:-down_proj}" \
    --n_samples "${N_SAMPLES:-64}" \
    --max_length "${MAX_LENGTH:-512}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --seed "${SEED:-42}"

echo ""
echo "Done: $(date)"
echo "Results in: $OUTPUT_DIR/"

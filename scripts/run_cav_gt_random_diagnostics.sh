#!/bin/bash
#SBATCH --job-name=cav_gt_rand_diag
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/cav_gt_rand_diag_%j.out
#SBATCH --error=logs/cav_gt_rand_diag_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started  : $(date)"
echo "========================================"

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

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/xie.yiyi/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/xie.yiyi/hf_cache/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CAV_MASK="${CAV_MASK:-/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt}"
GT_MASK="${GT_MASK:-/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt}"
RANDOM_MASK="${RANDOM_MASK:-/scratch/xie.yiyi/rl_casino_masks/llama8b/random_baseline_dpo_sp97.5_seed42.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/xie.yiyi/cav_gt_random_diagnostics_dpo}"
N_SAMPLES="${N_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-42}"

echo "[config] MODEL=$MODEL"
echo "[config] CAV_MASK=$CAV_MASK"
echo "[config] GT_MASK=$GT_MASK"
echo "[config] RANDOM_MASK=$RANDOM_MASK"
echo "[config] OUTPUT_DIR=$OUTPUT_DIR"
echo "[config] N_SAMPLES=$N_SAMPLES"
echo "[config] BATCH_SIZE=$BATCH_SIZE"

"$PYTHON_BIN" src/analysis/cav_gt_random_diagnostics.py \
    --model "$MODEL" \
    --cav_mask "$CAV_MASK" \
    --gt_mask "$GT_MASK" \
    --random_mask "$RANDOM_MASK" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --erank_device cuda

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR/"
echo "  - cav_gt_random_layer_diagnostics.csv"
echo "  - summary.json"
echo "  - erank_by_mask.json"
echo "  - cka_cav_vs_gt.json"
echo "  - cka_random_vs_gt.json"
echo "  - cka_cav_vs_random.json"
echo "========================================"

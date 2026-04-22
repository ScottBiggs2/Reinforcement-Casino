#!/bin/bash
#SBATCH --job-name=probe_ckpts_grpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/probe_ckpts_grpo_%j.out
#SBATCH --error=logs/probe_ckpts_grpo_%j.err

# Phase-1 probe over TRAINED checkpoints (scheme D):
#   - Baseline:        Llama-3.1-8B-Instruct (untrained reference)
#   - Dense-GRPO:      all-weights GRPO on OpenR1-Math-220k, step 200
#   - Sparse-GRPO:     magnitude-mask (Oracle) GRPO, step 200
#
# 6 probe properties: 4 benchmarks + tulu3 dpo_pref + openr1 grpo_pref.
# Strong L2 (probe_C=0.1) + 60/40 train/holdout split to prevent overfitting.

set -euo pipefail

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started  : $(date)"
echo "========================================"

CUDA_LIB=$(dirname "$(find /usr /opt /lib64 /lib -name "libcuda.so.1" 2>/dev/null | head -n1)")
if [ -n "$CUDA_LIB" ] && [ "$CUDA_LIB" != "." ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[gpu] injected libcuda from: $CUDA_LIB"
fi

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PRIMARY="/scratch/xie.yiyi/conda_envs/rl_casino"
CONDA_ENV_FALLBACK="/home/xie.yiyi/.conda/envs/rl_casino"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    if [ -d "$CONDA_ENV_PRIMARY" ]; then
        conda activate "$CONDA_ENV_PRIMARY"
    elif [ -d "$CONDA_ENV_FALLBACK" ]; then
        conda activate "$CONDA_ENV_FALLBACK"
    fi
    echo "[env] Python: $(which python)"
fi

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export HF_HOME="${HF_HOME:-/scratch/xie.yiyi/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/xie.yiyi/hf_cache/datasets}"
mkdir -p "$HF_DATASETS_CACHE"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/xie.yiyi/probe_ckpts_grpo_phase1}"
mkdir -p "$OUTPUT_DIR"

DENSE_GRPO="/scratch/xie.yiyi/rl_casino_outputs/llama8b_dpo_mask_grpo_dense_baseline_1gpu_20260408_044634/checkpoints/meta_llama_llama_3_1_8b_instruct_math_220k_grpo_dense/checkpoint-200"
SPARSE_GRPO_ORACLE="/scratch/xie.yiyi/rl_casino_outputs/llama8b_dpo_mask_grpo_sparse_magnitude_1gpu_20260409_031402/llama8b_dpo_mask_grpo_sparse_magnitude_1gpu_20260409_031402/final_model"

CKPTS_JSON="$OUTPUT_DIR/ckpts.json"
cat > "$CKPTS_JSON" <<JSON
[
  {"label": "Dense-GRPO (step200)",         "path": "$DENSE_GRPO"},
  {"label": "Sparse-GRPO-Oracle (step200)", "path": "$SPARSE_GRPO_ORACLE"}
]
JSON
echo "[config] Wrote checkpoint list: $CKPTS_JSON"
echo "[config] OUTPUT_DIR=$OUTPUT_DIR"

python src/analysis/probe_checkpoints.py \
    --ckpts_json "$CKPTS_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --layer_stride "${LAYER_STRIDE:-4}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --max_length "${MAX_LENGTH:-384}" \
    --cv_folds "${CV_FOLDS:-5}" \
    --pairs_per_pos "${PAIRS_PER_POS:-2}" \
    --n_jobs "${N_JOBS:-8}" \
    --probe_C "${PROBE_C:-0.1}" \
    --preference_samples_per_class "${PREFERENCE_SAMPLES_PER_CLASS:-1500}" \
    --holdout_frac "${HOLDOUT_FRAC:-0.4}" \
    --use_holdout_as_test

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR/"
echo "  - probe_results.json"
echo "  - probe_heatmap_all.png"
echo "  - probe_delta_all.png"
echo "========================================"

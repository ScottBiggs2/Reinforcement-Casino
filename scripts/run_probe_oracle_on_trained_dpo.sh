#!/bin/bash
#SBATCH --job-name=probe_oracle_trained_dpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/probe_oracle_trained_dpo_%j.out
#SBATCH --error=logs/probe_oracle_trained_dpo_%j.err

# Strict-matched probe: oracle mask applied to its OWN trained DPO model.
# No cross-model / cross-mask. Runs lightr1 and tulu3 back-to-back.

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

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

OUT_BASE="${OUT_BASE:-/scratch/xie.yiyi/probe_oracle_on_trained_dpo}"
mkdir -p "$OUT_BASE"

LAYER_STRIDE="${LAYER_STRIDE:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CV_FOLDS="${CV_FOLDS:-5}"
PAIRS_PER_POS="${PAIRS_PER_POS:-2}"
N_JOBS="${N_JOBS:-8}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"

run_pair() {
    local tag="$1"
    local model="$2"
    local mask="$3"
    local mask_label="$4"
    local out_dir="${OUT_BASE}/${tag}"

    echo ""
    echo "========================================"
    echo "[${tag}] MODEL = $model"
    echo "[${tag}] MASK  = $mask  (label=$mask_label)"
    echo "[${tag}] OUT   = $out_dir"
    echo "========================================"

    if [ ! -d "$model" ] && [ ! -f "$model/config.json" ]; then
        if [ ! -f "$model" ]; then
            echo "[${tag}] ERROR: model path not found"
            return 1
        fi
    fi
    if [ ! -f "$mask" ]; then
        echo "[${tag}] ERROR: mask file not found: $mask"
        return 1
    fi

    mkdir -p "$out_dir"
    local masks_json="${out_dir}/oracle_only.json"
    cat > "$masks_json" <<JSON
[
  {"label": "${mask_label}", "path": "${mask}"}
]
JSON

    "$PYTHON_BIN" src/analysis/probe_pair_masks.py \
        --model "$model" \
        --masks_json "$masks_json" \
        --output_dir "$out_dir" \
        --layer_stride "$LAYER_STRIDE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --cv_folds "$CV_FOLDS" \
        --pairs_per_pos "$PAIRS_PER_POS" \
        --n_jobs "$N_JOBS" \
        --holdout_frac "$HOLDOUT_FRAC" \
        --use_holdout_as_test
}

# === A. light_r1 ============================================================
run_pair \
    "lightr1" \
    "/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500" \
    "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step500_sp97.5.pt" \
    "Oracle-DPO-lightr1-step500"

# === B. tulu3 ===============================================================
run_pair \
    "tulu3" \
    "/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500" \
    "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_tulu3_step500_sp97.5.pt" \
    "Oracle-DPO-tulu3-step500"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results under: $OUT_BASE/{lightr1,tulu3}/"
echo "  - probe_pair_results.json"
echo "  - probe_pair_heatmap_all.png   (dense baseline + oracle subnetwork)"
echo "  - probe_pair_delta_all.png     (oracle − baseline)"
echo "========================================"

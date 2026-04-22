#!/bin/bash
#SBATCH --job-name=probe_pair_C
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/probe_pair_scheme_c_%j.out
#SBATCH --error=logs/probe_pair_scheme_c_%j.err

# Scheme C probe run: 4 generic benchmarks + 2 preference probes
#   (tulu3 dpo_preference, OpenR1-Math-220k grpo_preference).
# Uses stronger L2 (probe_C=0.1) and a holdout split to guard against the
# p>>n overfitting seen on the 6-mask runs.

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
export HF_HOME="${HF_HOME:-/scratch/xie.yiyi/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/xie.yiyi/hf_cache/datasets}"
mkdir -p "$HF_DATASETS_CACHE"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/xie.yiyi/probe_pair_scheme_c}"
mkdir -p "$OUTPUT_DIR"

RANDOM_SEED="${RANDOM_SEED:-42}"
SPARSITY="${SPARSITY:-97.5}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

DPO_ORACLE="/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt"
GRPO_ORACLE="/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt"
DPO_RANDOM="/scratch/xie.yiyi/rl_casino_masks/llama8b/random_baseline_dpo_sp97.5_seed${RANDOM_SEED}.pt"
GRPO_RANDOM="/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/random_baseline_grpo_sp97.5_seed${RANDOM_SEED}.pt"
DPO_CAV="/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt"
GRPO_CAV="/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_grpo.pt"

# Regenerate random baselines only if the expected mask files are present
# but random files are missing. Safe to re-run — compares to reference when done.
generate_random_if_missing() {
    local ref_mask="$1"
    local out_mask="$2"
    local label="$3"
    if [ -s "$out_mask" ]; then
        echo "[random] OK: $label exists at $out_mask"
        return
    fi
    if [ ! -s "$ref_mask" ]; then
        echo "[random] FATAL: reference mask missing for $label: $ref_mask"
        exit 1
    fi
    echo "[random] Generating $label"
    mkdir -p "$(dirname "$out_mask")"
    "$PYTHON_BIN" src/warm_start/random_mask_baseline.py \
        --reference_mask "$ref_mask" \
        --sparsity_percent "$SPARSITY" \
        --output_file "$out_mask" \
        --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
        --seed "$RANDOM_SEED" \
        --compare_to_reference
}

generate_random_if_missing "$DPO_ORACLE"  "$DPO_RANDOM"  "Random-DPO"
generate_random_if_missing "$GRPO_ORACLE" "$GRPO_RANDOM" "Random-GRPO"

MASKS_JSON="${MASKS_JSON:-$OUTPUT_DIR/masks_scheme_c.json}"
if [ ! -f "$MASKS_JSON" ] || [ -n "${FORCE_REGEN_MASKS_JSON:-}" ]; then
    cat > "$MASKS_JSON" <<JSON
[
  {"label": "Random-DPO",   "path": "$DPO_RANDOM"},
  {"label": "Cold-CAV-DPO", "path": "$DPO_CAV"},
  {"label": "Oracle-DPO",   "path": "$DPO_ORACLE"},
  {"label": "Random-GRPO",  "path": "$GRPO_RANDOM"},
  {"label": "Cold-CAV-GRPO","path": "$GRPO_CAV"},
  {"label": "Oracle-GRPO",  "path": "$GRPO_ORACLE"}
]
JSON
    echo "[config] Wrote mask list: $MASKS_JSON"
fi

echo "[config] MODEL=$MODEL"
echo "[config] OUTPUT_DIR=$OUTPUT_DIR"
echo "[config] MASKS_JSON=$MASKS_JSON"
echo "[config] PROBE_C=${PROBE_C:-0.1}"
echo "[config] PREFERENCE_SAMPLES_PER_CLASS=${PREFERENCE_SAMPLES_PER_CLASS:-1000}"
echo "[config] HOLDOUT_FRAC=${HOLDOUT_FRAC:-0.2}"
echo "[config] USE_HOLDOUT_AS_TEST=${USE_HOLDOUT_AS_TEST:-1}"

EXTRA_ARGS=()
if [ "${USE_HOLDOUT_AS_TEST:-1}" = "1" ]; then
    EXTRA_ARGS+=(--use_holdout_as_test)
fi

"$PYTHON_BIN" src/analysis/probe_pair_masks.py \
    --model "$MODEL" \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --layer_stride "${LAYER_STRIDE:-4}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --max_length "${MAX_LENGTH:-384}" \
    --cv_folds "${CV_FOLDS:-5}" \
    --pairs_per_pos "${PAIRS_PER_POS:-2}" \
    --n_jobs "${N_JOBS:-8}" \
    --probe_C "${PROBE_C:-0.1}" \
    --preference_samples_per_class "${PREFERENCE_SAMPLES_PER_CLASS:-1000}" \
    --holdout_frac "${HOLDOUT_FRAC:-0.2}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR/"
echo "  - probe_pair_results.json"
echo "  - probe_pair_heatmap_all.png"
echo "  - probe_pair_delta_all.png"
echo "========================================"

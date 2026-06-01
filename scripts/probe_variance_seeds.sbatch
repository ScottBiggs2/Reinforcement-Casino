#!/bin/bash
#SBATCH --job-name=probe_variance_seeds
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/probe_variance_seeds_%j.out
#SBATCH --error=logs/probe_variance_seeds_%j.err

# Multi-seed pairwise-ranking probe variance study.
#   Mask seeds: 42, 43, 44   (Cold-CAV + Random regen per seed; Oracle reused.)
#   Probe seed: fixed at 42 (so variance reflects mask construction only).
#
# Output:
#   /scratch/$USER/probe_variance/masks/   regenerated CAV + Random masks
#   /scratch/$USER/probe_variance/seed42/  probe_pair_results.json + plots
#   /scratch/$USER/probe_variance/seed43/  ...
#   /scratch/$USER/probe_variance/seed44/  ...

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
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export HF_DATASETS_CACHE="/scratch/xie.yiyi/hf_cache/datasets"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SPARSITY="${SPARSITY:-97.5}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
SEEDS="${SEEDS:-42 43 44}"

# Shared oracles (deterministic — reused across seeds)
DPO_ORACLE="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step500_sp97.5.pt"
GRPO_ORACLE="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt"

VAR_ROOT="/scratch/xie.yiyi/probe_variance"
MASK_DIR="$VAR_ROOT/masks"
mkdir -p "$MASK_DIR"

if [ ! -s "$DPO_ORACLE" ] || [ ! -s "$GRPO_ORACLE" ]; then
    echo "FATAL: Oracle mask missing."
    echo "  DPO oracle:  $DPO_ORACLE  $([ -s "$DPO_ORACLE" ] && echo OK || echo MISSING)"
    echo "  GRPO oracle: $GRPO_ORACLE $([ -s "$GRPO_ORACLE" ] && echo OK || echo MISSING)"
    exit 1
fi

echo "[config] MODEL=$MODEL  SPARSITY=$SPARSITY  SEEDS=$SEEDS"
echo "[config] DPO_ORACLE=$DPO_ORACLE"
echo "[config] GRPO_ORACLE=$GRPO_ORACLE"

generate_cav() {
    local mode="$1" seed="$2" out="$3"
    if [ -s "$out" ]; then
        echo "[cav] OK: $mode seed=$seed exists at $out"
        return
    fi
    echo "[cav] Generating $mode seed=$seed"
    local CAV_EXTRA=()
    if python src/cold_start/inference_mask_finder.py --help 2>&1 | grep -q -- "--min_layer_keep_ratio"; then
        CAV_EXTRA+=(--min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO")
    fi
    python src/cold_start/inference_mask_finder.py \
        --model_name "$MODEL" \
        --method cav \
        --mode "$mode" \
        --n_samples 256 \
        --max_length 256 \
        --batch_size 8 \
        --seed "$seed" \
        --sparsity "$SPARSITY" \
        "${CAV_EXTRA[@]}" \
        --verbose \
        --output "$out"
}

generate_random() {
    local ref="$1" seed="$2" out="$3" label="$4"
    if [ -s "$out" ]; then
        echo "[random] OK: $label seed=$seed exists at $out"
        return
    fi
    echo "[random] Generating $label seed=$seed"
    python src/warm_start/random_mask_baseline.py \
        --reference_mask "$ref" \
        --sparsity_percent "$SPARSITY" \
        --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
        --seed "$seed" \
        --output_file "$out" \
        --compare_to_reference
}

for SEED in $SEEDS; do
    echo ""
    echo "================ SEED $SEED ================"

    CAV_DPO="$MASK_DIR/cold_cav_v2_dpo_seed${SEED}.pt"
    CAV_GRPO="$MASK_DIR/cold_cav_v2_grpo_seed${SEED}.pt"
    RAND_DPO="$MASK_DIR/random_lightr1_sp97.5_seed${SEED}.pt"
    RAND_GRPO="$MASK_DIR/random_grpo_math220k_sp97.5_seed${SEED}.pt"

    generate_cav    dpo  "$SEED" "$CAV_DPO"
    generate_cav    grpo "$SEED" "$CAV_GRPO"
    generate_random "$DPO_ORACLE"  "$SEED" "$RAND_DPO"  "Random-DPO"
    generate_random "$GRPO_ORACLE" "$SEED" "$RAND_GRPO" "Random-GRPO"

    OUT_DIR="$VAR_ROOT/seed${SEED}"
    mkdir -p "$OUT_DIR"
    MASKS_JSON="$OUT_DIR/masks_seed${SEED}.json"
    cat > "$MASKS_JSON" <<JSON
[
  {"label": "Random-DPO",    "path": "$RAND_DPO"},
  {"label": "Cold-CAV-DPO",  "path": "$CAV_DPO"},
  {"label": "Oracle-DPO",    "path": "$DPO_ORACLE"},
  {"label": "Random-GRPO",   "path": "$RAND_GRPO"},
  {"label": "Cold-CAV-GRPO", "path": "$CAV_GRPO"},
  {"label": "Oracle-GRPO",   "path": "$GRPO_ORACLE"}
]
JSON

    if [ -s "$OUT_DIR/probe_pair_results.json" ]; then
        echo "[probe] OK: seed=$SEED results already at $OUT_DIR/probe_pair_results.json"
        continue
    fi

    echo "[probe] Running probe pass for seed=$SEED"
    python src/analysis/probe_pair_masks.py \
        --model "$MODEL" \
        --masks_json "$MASKS_JSON" \
        --output_dir "$OUT_DIR" \
        --layer_stride "${LAYER_STRIDE:-4}" \
        --batch_size "${BATCH_SIZE:-8}" \
        --max_length "${MAX_LENGTH:-256}" \
        --cv_folds "${CV_FOLDS:-5}" \
        --pairs_per_pos "${PAIRS_PER_POS:-2}" \
        --n_jobs "${N_JOBS:-8}" \
        --holdout_frac "${HOLDOUT_FRAC:-0.2}" \
        --no_preference \
        --use_holdout_as_test
done

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Per-seed outputs under: $VAR_ROOT/"
for s in $SEEDS; do
    echo "  seed=$s : $VAR_ROOT/seed${s}/probe_pair_results.json"
done
echo "========================================"

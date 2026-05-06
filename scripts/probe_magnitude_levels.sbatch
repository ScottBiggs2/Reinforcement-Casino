#!/bin/bash
#SBATCH --job-name=probe_mag_levels
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d1025
#SBATCH --time=01:30:00
#SBATCH --output=logs/probe_mag_levels_%j.out
#SBATCH --error=logs/probe_mag_levels_%j.err

# ---------------------------------------------------------------------------
# Probe (zero_out only) of warm-magnitude masks at target_step ∈ {50, 100, 200}
# vs. oracle (checkpoint-diff), random, dense baseline. Done for both
# tulu3 and lightr1 dense DPO ckpt-500 models. --no_preference (4
# benchmark probes only).
# ---------------------------------------------------------------------------

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
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate /home/xie.yiyi/.conda/envs/rl_casino
fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export RL_CASINO_WARM_MASK_SCORE_DEVICE="${RL_CASINO_WARM_MASK_SCORE_DEVICE:-cuda}"

WARM_DIR="/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b"
ORACLE_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
mkdir -p "$WARM_DIR"

DELTA_TULU3="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b_deltalog/deltas/meta_llama_llama_3_1_8b_instruct_tulu3"
DELTA_LIGHTR1="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/deltas/meta_llama_llama_3_1_8b_instruct_light_r1"

DENSE_TULU3="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
DENSE_LIGHTR1="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500"

SPARSITY=97.5
MIN_LAYER_KEEP_RATIO=0.0025

# ---------------------------------------------------------------------------
# Step 1: generate any missing magnitude masks (target_step in 50, 100, 200)
# ---------------------------------------------------------------------------
gen_mag() {
    local tag="$1"        # tulu3 | lightr1
    local delta_dir="$2"
    local target="$3"
    local out="$WARM_DIR/warm_magnitude_dpo_${tag}_step${target}_sp${SPARSITY}.pt"
    if [ -f "$out" ]; then
        echo "[mask] exists: $out"
        return
    fi
    echo "[mask] generating warm-magnitude $tag step=$target"
    python src/warm_start/even_better_mask_finder.py \
        --delta_log_dir "$delta_dir" \
        --method magnitude \
        --sparsity_percent "$SPARSITY" \
        --target_step "$target" \
        --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
        --output_file "$out"
}

for s in 50 100 200; do
    gen_mag tulu3   "$DELTA_TULU3"   "$s"
    gen_mag lightr1 "$DELTA_LIGHTR1" "$s"
done

# ---------------------------------------------------------------------------
# Step 2: write masks JSONs (one per dataset)
# ---------------------------------------------------------------------------
OUT_BASE="/scratch/xie.yiyi/probe_magnitude_levels"
mkdir -p "$OUT_BASE/tulu3" "$OUT_BASE/lightr1"

write_masks_json() {
    local tag="$1"
    local out_json="$OUT_BASE/$tag/masks.json"
    cat > "$out_json" <<JSON
[
  {"label": "Oracle",         "path": "$ORACLE_DIR/oracle_dpo_${tag}_step500_sp${SPARSITY}.pt"},
  {"label": "Random-seed42",  "path": "$ORACLE_DIR/random_baseline_${tag}_sp${SPARSITY}_seed42.pt"},
  {"label": "Magnitude-50",   "path": "$WARM_DIR/warm_magnitude_dpo_${tag}_step50_sp${SPARSITY}.pt"},
  {"label": "Magnitude-100",  "path": "$WARM_DIR/warm_magnitude_dpo_${tag}_step100_sp${SPARSITY}.pt"},
  {"label": "Magnitude-200",  "path": "$WARM_DIR/warm_magnitude_dpo_${tag}_step200_sp${SPARSITY}.pt"}
]
JSON
    echo "[main] wrote $out_json:"
    cat "$out_json"
}

write_masks_json tulu3
write_masks_json lightr1

# ---------------------------------------------------------------------------
# Step 3: probe each dataset (zero_out mode only, 4 benchmark probes)
# ---------------------------------------------------------------------------
run_probe() {
    local tag="$1"
    local model="$2"
    local out_dir="$OUT_BASE/$tag/zero_out"
    mkdir -p "$out_dir"

    echo ""
    echo "========================================"
    echo "[$tag] probe MODEL = $model"
    echo "[$tag] OUT          = $out_dir"
    echo "========================================"

    python src/analysis/probe_pair_masks.py \
        --model "$model" \
        --masks_json "$OUT_BASE/$tag/masks.json" \
        --output_dir "$out_dir" \
        --layer_stride 4 \
        --batch_size 8 \
        --max_length 256 \
        --cv_folds 3 \
        --pairs_per_pos 2 \
        --n_jobs 8 \
        --holdout_frac 0.2 \
        --use_holdout_as_test \
        --probe_C 0.01 \
        --no_preference \
        --patch_mode zero_out
}

run_probe tulu3   "$DENSE_TULU3"
run_probe lightr1 "$DENSE_LIGHTR1"

echo ""
echo "Done: $(date)"
echo "Results:"
echo "  $OUT_BASE/{tulu3,lightr1}/zero_out/probe_pair_results.json"

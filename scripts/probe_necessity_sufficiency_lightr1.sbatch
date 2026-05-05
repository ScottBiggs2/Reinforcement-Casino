#!/bin/bash
#SBATCH --job-name=probe_necsuff_lr1
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d1025
#SBATCH --time=01:30:00
#SBATCH --output=logs/probe_necsuff_lr1_%j.out
#SBATCH --error=logs/probe_necsuff_lr1_%j.err

# ---------------------------------------------------------------------------
# Necessity-Sufficiency probe study — LIGHT-R1 mirror of the tulu3 run
# ---------------------------------------------------------------------------
# Identical protocol as scripts/probe_necessity_sufficiency.sbatch (job
# 6526239 lineage), but:
#   - Oracle / Anti-Oracle / Random masks are matched to the lightr1 oracle
#     (97.5% sparsity, lightr1 ckpt-500 - base diff)
#   - Probed model is the dense lightr1 DPO ckpt-500
#   - --no_preference: skip dpo_preference and grpo_preference probes
#     (those were below-chance even on the unmasked baseline in the tulu3
#     study, indicating the linear probe pipeline can't read those signals
#     at this configuration). Only run the 4 benchmark probes:
#     syntax / semantics / factual / math.
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

# ---------------------------------------------------------------------------
# Paths (lightr1)
# ---------------------------------------------------------------------------
OUT_BASE="/scratch/xie.yiyi/probe_necsuff_lightr1"
MASK_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
ORACLE_MASK="$MASK_DIR/oracle_dpo_lightr1_step500_sp97.5.pt"
ANTI_MASK="$MASK_DIR/anti_oracle_dpo_lightr1_step500_sp97.5.pt"
DENSE_MODEL="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Existing seed42 + 2 new seeds (43, 44) for the random control band.
RANDOM_SEEDS=(42 43 44)

if [ ! -f "$ORACLE_MASK" ]; then echo "[error] oracle mask missing: $ORACLE_MASK"; exit 1; fi
if [ ! -d "$DENSE_MODEL"  ]; then echo "[error] dense model missing: $DENSE_MODEL"; exit 1; fi

mkdir -p "$OUT_BASE"

# ---------------------------------------------------------------------------
# Step 1: build anti-mask (1 - oracle) if missing
# ---------------------------------------------------------------------------
if [ ! -f "$ANTI_MASK" ]; then
    echo "[mask] generating anti-mask -> $ANTI_MASK"
    python src/warm_start/anti_mask.py \
        --reference_mask "$ORACLE_MASK" \
        --output_file    "$ANTI_MASK"
else
    echo "[mask] anti-mask exists: $ANTI_MASK"
fi

# ---------------------------------------------------------------------------
# Step 2: build random masks for missing seeds (parallel; 2 procs × ~80 GB
# RSS each fits in --mem=72G... actually no, 2 × 80 = 160 GB > 72 GB. We
# only need to gen if seed42 doesn't exist OR seed43/44 don't. Existing
# seed42 is on disk for lightr1. To stay within --mem=72G, generate
# sequentially when missing — only ever 2 to gen at most, ~50 min wall.)
# ---------------------------------------------------------------------------
for seed in "${RANDOM_SEEDS[@]}"; do
    rmask="$MASK_DIR/random_baseline_lightr1_sp97.5_seed${seed}.pt"
    if [ -f "$rmask" ]; then
        echo "[mask] random seed=${seed} exists"
        continue
    fi
    echo "[mask] generating random seed=${seed} -> $rmask"
    python src/warm_start/random_mask_baseline.py \
        --reference_mask  "$ORACLE_MASK" \
        --sparsity_percent 97.5 \
        --output_file     "$rmask" \
        --seed            "$seed"
done

# ---------------------------------------------------------------------------
# Step 3: write the masks JSON
# ---------------------------------------------------------------------------
MASKS_JSON="$OUT_BASE/masks.json"
{
    echo '['
    echo "  {\"label\": \"Oracle\",      \"path\": \"${ORACLE_MASK}\"},"
    echo "  {\"label\": \"Anti-Oracle\", \"path\": \"${ANTI_MASK}\"},"
    for i in "${!RANDOM_SEEDS[@]}"; do
        seed="${RANDOM_SEEDS[$i]}"
        rmask="$MASK_DIR/random_baseline_lightr1_sp97.5_seed${seed}.pt"
        last=$(( i + 1 == ${#RANDOM_SEEDS[@]} ))
        comma=","
        [ "$last" -eq 1 ] && comma=""
        echo "  {\"label\": \"Random-seed${seed}\", \"path\": \"${rmask}\"}${comma}"
    done
    echo ']'
} > "$MASKS_JSON"
echo "[main] wrote $MASKS_JSON"
cat "$MASKS_JSON"

# ---------------------------------------------------------------------------
# Step 4: run probe in all three patch modes — benchmark probes only
# (--no_preference). On h200 each mode takes ~7 min (4 probes × 6 configs).
# ---------------------------------------------------------------------------
run_mode() {
    local mode="$1"
    local out_dir="$OUT_BASE/$mode"
    mkdir -p "$out_dir"

    echo ""
    echo "========================================"
    echo "[$mode] OUT = $out_dir"
    echo "========================================"

    local extra=""
    if [ "$mode" != "zero_out" ]; then
        extra="--base_model $BASE_MODEL"
    fi

    python src/analysis/probe_pair_masks.py \
        --model "$DENSE_MODEL" \
        --masks_json "$MASKS_JSON" \
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
        --patch_mode "$mode" \
        $extra
}

run_mode "zero_out"
run_mode "delta_only"
run_mode "anti_delta_only"

echo ""
echo "Done: $(date)"
echo "Results:"
echo "  $OUT_BASE/{zero_out,delta_only,anti_delta_only}/probe_pair_results.json"

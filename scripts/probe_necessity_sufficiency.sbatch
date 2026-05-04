#!/bin/bash
#SBATCH --job-name=probe_necsuff
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025
#SBATCH --time=02:30:00
#SBATCH --output=logs/probe_necsuff_%j.out
#SBATCH --error=logs/probe_necsuff_%j.err

# ---------------------------------------------------------------------------
# Necessity-Sufficiency probe study for tulu3 DPO oracle subnetwork
# ---------------------------------------------------------------------------
# Three legs prove "M identifies the task subnetwork":
#   1. Oracle  (keep M)            -> probe signal preserved   (sufficiency)
#   2. Anti-Oracle (keep ¬M = 1-M) -> probe signal destroyed   (necessity)
#   3. Random x6 seeds, matched per-layer density               (specificity)
#
# Two patch modes:
#   * zero_out          (legacy): w' = w_ft * M
#   * anti_delta_only   (LTH):    w' = w_base + Δθ * (1-M)  -- delta on ¬M
#   * delta_only        (LTH):    w' = w_base + Δθ * M      -- delta on M only
# This sbatch runs ALL three modes against the dense tulu3 ckpt-500 model.
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
# Paths
# ---------------------------------------------------------------------------
OUT_BASE="/scratch/xie.yiyi/probe_necsuff_tulu3"
MASK_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
ORACLE_MASK="$MASK_DIR/oracle_dpo_tulu3_step500_sp97.5.pt"
ANTI_MASK="$MASK_DIR/anti_oracle_dpo_tulu3_step500_sp97.5.pt"
DENSE_MODEL="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Existing seed42 + 2 new seeds for the random control band.
# Why only 3: random_mask_baseline.py uses ~80 GB RSS per process at 8B
# params (global score tensor + sort buffers). 5 parallel × 80 GB blew
# past --mem=320G into swap. Keeping only 2 to gen (43, 44) fits in
# 2×80=160 GB with comfortable headroom. random×3 still gives a valid
# (if wider-CI) variance estimate.
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
# Step 2: build random masks for missing seeds (PARALLEL — each gen is
# ~25 min serial on 8B params via chunked global top-k; per-process RSS
# ~32 GB. Running them concurrently drops wall time from ~125 min to
# ~25 min. We allocated 320 GB / 16 CPUs so 5×32 GB and 5×1 CPU fit.)
# ---------------------------------------------------------------------------
RAND_PIDS=()
for seed in "${RANDOM_SEEDS[@]}"; do
    rmask="$MASK_DIR/random_baseline_tulu3_sp97.5_seed${seed}.pt"
    if [ -f "$rmask" ]; then
        echo "[mask] random seed=${seed} exists"
        continue
    fi
    echo "[mask] launching random seed=${seed} -> $rmask (background)"
    python src/warm_start/random_mask_baseline.py \
        --reference_mask  "$ORACLE_MASK" \
        --sparsity_percent 97.5 \
        --output_file     "$rmask" \
        --seed            "$seed" \
        > "logs/random_mask_seed${seed}_${SLURM_JOB_ID:-local}.log" 2>&1 &
    RAND_PIDS+=($!)
done

if [ ${#RAND_PIDS[@]} -gt 0 ]; then
    echo "[mask] waiting on ${#RAND_PIDS[@]} parallel mask gen(s): ${RAND_PIDS[*]}"
    fail=0
    for pid in "${RAND_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "[mask] FAILED pid $pid"
            fail=1
        fi
    done
    if [ "$fail" -eq 1 ]; then
        echo "[mask] one or more random mask gens failed; see logs/random_mask_seed*_${SLURM_JOB_ID:-local}.log"
        exit 1
    fi
    echo "[mask] all parallel mask gens completed"
fi

# ---------------------------------------------------------------------------
# Step 3: write the masks JSON (1 oracle + 1 anti + 6 random)
# ---------------------------------------------------------------------------
MASKS_JSON="$OUT_BASE/masks.json"
{
    echo '['
    echo "  {\"label\": \"Oracle\",      \"path\": \"${ORACLE_MASK}\"},"
    echo "  {\"label\": \"Anti-Oracle\", \"path\": \"${ANTI_MASK}\"},"
    for i in "${!RANDOM_SEEDS[@]}"; do
        seed="${RANDOM_SEEDS[$i]}"
        rmask="$MASK_DIR/random_baseline_tulu3_sp97.5_seed${seed}.pt"
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
# Step 4: run probe in all three patch modes
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

    # Protocol: 3-fold CV + 20% holdout, L2 C=0.01.
    # Why CV came back: cv_folds=0 (single 80/20 split) gave Baseline
    # preference probe acc=0.355/0.176 — below chance — even though the
    # 4 benchmark probes scored 0.92-0.99. The single split is too
    # high-variance on the 1270-sample × 14336-dim preference probes
    # (job 6522862 confirmed). 3-fold CV averages out the unlucky split
    # without re-introducing 5-fold reporting clutter; holdout is still
    # reported as the primary test metric, CV is the stabilizer.
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

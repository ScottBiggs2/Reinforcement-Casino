#!/bin/bash
#SBATCH --job-name=necsuff_lr1_v2
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=d1025
#SBATCH --time=03:00:00
#SBATCH --output=logs/necsuff_lr1_v2_%j.out
#SBATCH --error=logs/necsuff_lr1_v2_%j.err

# ---------------------------------------------------------------------------
# A: Necessity-Sufficiency probe study — Light-R1, v2 (rebuttal rerun)
# ---------------------------------------------------------------------------
# Differences vs scripts/probe_necessity_sufficiency_lightr1.sbatch (never run):
#   1. ORACLE_MASK -> the SELF-CONSISTENT `_src-deltafull` mask built by
#      8817043 from the 07-28 dense retrain. The v1 script pointed at the
#      orphaned Apr-27 mask while probing the retrained checkpoint, which
#      would conflate run-to-run mask drift with the patching effect.
#   2. Random band -> matched `random_matched_*` masks (291 tensors incl.
#      layer-norms, built in graft_prep from the same oracle reference).
#      The old seed42 file covers 290 tensors with no layer-norms.
#   3. zero_out mode DROPPED: the zero_out record exists (May runs) and the
#      operator is destructive (measures base-model survival, not the
#      update). Only delta_only + anti_delta_only — the two legs the
#      rebuttal sentence needs. Saves ~0.9 GPU-h.
#   4. h200 -> a100 (pure 8B forwards at bs8/len256; h200s congested).
# Probe hyperparameters are byte-identical to the frozen 05-03 protocol.
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

OUT_BASE="/scratch/xie.yiyi/probe_necsuff_lightr1_v2"
MASK_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
ORACLE_MASK="$MASK_DIR/oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt"
ANTI_MASK="$MASK_DIR/anti_oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt"
DENSE_MODEL="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
RANDOM_SEEDS=(42 43 44)

# All masks are built upstream by graft_prep (afterok). Hard-fail if absent —
# regenerating here would race the prep job's skip-if-exists logic.
for f in "$ORACLE_MASK" "$ANTI_MASK"; do
    test -s "$f" || { echo "[error] missing mask: $f"; exit 1; }
done
for seed in "${RANDOM_SEEDS[@]}"; do
    f="$MASK_DIR/random_matched_lightr1_sp97.5_seed${seed}.pt"
    test -s "$f" || { echo "[error] missing matched random: $f"; exit 1; }
done
test -d "$DENSE_MODEL" || { echo "[error] dense model missing"; exit 1; }
mkdir -p "$OUT_BASE"

MASKS_JSON="$OUT_BASE/masks.json"
{
    echo '['
    echo "  {\"label\": \"Oracle\",      \"path\": \"${ORACLE_MASK}\"},"
    echo "  {\"label\": \"Anti-Oracle\", \"path\": \"${ANTI_MASK}\"},"
    for i in "${!RANDOM_SEEDS[@]}"; do
        seed="${RANDOM_SEEDS[$i]}"
        rmask="$MASK_DIR/random_matched_lightr1_sp97.5_seed${seed}.pt"
        comma=","
        [ $(( i + 1 )) -eq ${#RANDOM_SEEDS[@]} ] && comma=""
        echo "  {\"label\": \"Random-seed${seed}\", \"path\": \"${rmask}\"}${comma}"
    done
    echo ']'
} > "$MASKS_JSON"
echo "[main] wrote $MASKS_JSON"
cat "$MASKS_JSON"

run_mode() {
    local mode="$1"
    local out_dir="$OUT_BASE/$mode"
    mkdir -p "$out_dir"
    echo ""
    echo "========================================"
    echo "[$mode] OUT = $out_dir"
    echo "========================================"
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
        --base_model "$BASE_MODEL"
}

run_mode "delta_only"
run_mode "anti_delta_only"

echo ""
echo "Done: $(date)"
echo "Results: $OUT_BASE/{delta_only,anti_delta_only}/probe_pair_results.json"

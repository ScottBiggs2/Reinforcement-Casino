#!/bin/bash
# ============================================================================
# Block Size Ablation — Single H200 GPU
#
# Sweeps block_size_bsr over {8, 16, 32, 64} with fixed block_size_adam=128.
# Also sweeps block_size_adam over {64, 128, 256} with fixed block_size_bsr=16.
#
# Submit: sbatch scripts/ablation_block_size.sh
# Local:  bash scripts/ablation_block_size.sh
# ============================================================================
#SBATCH --job-name=bsr_ablation
#SBATCH --output=logs/bsr_ablation_%j.out
#SBATCH --error=logs/bsr_ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

# ── Repo root ───────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

echo "====== Block Size Ablation ======"
echo "Date:     $(date)"
echo "Node:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"

# ── Environment ─────────────────────────────────────────────────────────────
CONDA_ENV_PRIMARY="/scratch/xie.yiyi/conda_envs/rl_casino"
CONDA_ENV_FALLBACK="/home/xie.yiyi/.conda/envs/rl_casino"

if [ -d "$CONDA_ENV_PRIMARY" ]; then
    ENV_PATH="$CONDA_ENV_PRIMARY"
elif [ -d "$CONDA_ENV_FALLBACK" ]; then
    ENV_PATH="$CONDA_ENV_FALLBACK"
else
    echo "ERROR: conda env not found"; exit 1
fi

PYTHON="$ENV_PATH/bin/python"
export PATH="$ENV_PATH/bin:$PATH"
export PYTHONPATH=.

# ── Shared Config ───────────────────────────────────────────────────────────
MODEL="google/gemma-3-270m-it"
DATASET="math-220k"
N_STEPS=50
SUBSET=512
BATCH_SIZE=1
GRAD_ACCUM=8
NUM_GENERATIONS=8
GEN_BATCH_SIZE=8
LR=5e-6
OPTIMIZER="sparse_adamw"

MASK_PATH="masks/top_10.0pct_momentum_w25_step25.pt"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/scratch/biggs.s/rl_casino_outputs/ablation_blocksize_${TIMESTAMP}"
CACHE_DIR="/scratch/biggs.s/hf_cache/datasets"
mkdir -p "$OUTPUT_DIR"

PASS=0
FAIL=0
TOTAL=0

run_sparse() {
    local BSR_SIZE="$1"
    local ADAM_SIZE="$2"
    local LABEL="bsr${BSR_SIZE}_adam${ADAM_SIZE}"
    local RUN_NAME="ablation_${LABEL}"

    echo ""
    echo "============================================================"
    echo "[$((TOTAL+1))] block_size_bsr=$BSR_SIZE  block_size_adam=$ADAM_SIZE"
    echo "============================================================"
    TOTAL=$((TOTAL + 1))

    if "$PYTHON" src/full_training/sparse_grpo_bsr.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --n_steps "$N_STEPS" \
        --subset_size "$SUBSET" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GEN_BATCH_SIZE" \
        --lr "$LR" \
        --optimizer "$OPTIMIZER" \
        --mask "$MASK_PATH" \
        --block_size_bsr "$BSR_SIZE" \
        --block_size_adam "$ADAM_SIZE" \
        --output_base_dir "$OUTPUT_DIR" \
        --dataset_cache_dir "$CACHE_DIR" \
        --save_model false \
        --use_wandb \
        --run_name "$RUN_NAME"; then
        echo "  -> $LABEL: PASSED"
        PASS=$((PASS + 1))
    else
        echo "  -> $LABEL: FAILED"
        FAIL=$((FAIL + 1))
    fi
}

# ── Sweep 1: block_size_bsr with fixed block_size_adam=128 ───────────────────
echo ""
echo "=== Sweep: block_size_bsr in {8, 16, 32, 64}, block_size_adam=128 ==="
for BSR in 8 16 32 64; do
    run_sparse "$BSR" 128
done

# ── Sweep 2: block_size_adam with fixed block_size_bsr=16 ────────────────────
echo ""
echo "=== Sweep: block_size_adam in {64, 256}, block_size_bsr=16 ==="
# block_size_adam=128 already covered above
for ADAM in 64 256; do
    run_sparse 16 "$ADAM"
done

# ── Collect Results ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Collecting ablation results..."
echo "============================================================"

"$PYTHON" -c "
import json, glob, os

out_dir = '$OUTPUT_DIR'
results = []
for f in sorted(glob.glob(os.path.join(out_dir, '*/timing_results.json'))):
    with open(f) as fh:
        d = json.load(fh)
        results.append(d)
        bsr = d.get('block_size_bsr', '?')
        adam = d.get('block_size_adam', '?')
        wall = d.get('wall_time', 0)
        per_step = d.get('time_per_step_wall', 0)
        gpu_time = d.get('gpu_time', 0)
        gpu_per = d.get('time_per_step_gpu', 0)
        print(f'  bsr={bsr:>3}  adam={adam:>3}  wall={wall:7.1f}s  step={per_step:5.2f}s  gpu={gpu_time:7.1f}s  gpu/step={gpu_per:5.2f}s')

summary_path = os.path.join(out_dir, 'ablation_summary.json')
with open(summary_path, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'\nSummary saved to {summary_path}')
"

echo ""
echo "============================================================"
echo "ABLATION COMPLETE"
echo "============================================================"
echo "Passed: $PASS / $TOTAL"
echo "Failed: $FAIL / $TOTAL"
echo "Output: $OUTPUT_DIR"
echo "Finished: $(date)"

[ "$FAIL" -gt 0 ] && exit 1 || exit 0

#!/bin/bash
#SBATCH --job-name=probe_lightr1_mask_on_allen
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/probe_lightr1_mask_on_allen_%j.out
#SBATCH --error=logs/probe_lightr1_mask_on_allen_%j.err

# Cross-trajectory probe: apply the user's lightr1 oracle mask AND a matched
# random mask to the public allenai/Llama-3.1-Tulu-3-8B-DPO checkpoint.
# Compares: dense baseline / oracle-masked / random-masked.
# Goal: see if oracle mask preserves significantly more probe accuracy than
# a random mask of the same sparsity (97.5%).

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

MODEL="allenai/Llama-3.1-Tulu-3-8B-DPO"
OUT_DIR="/scratch/xie.yiyi/probe_lightr1_mask_on_allen_tulu3"
MASK_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
ORACLE_MASK="$MASK_DIR/oracle_dpo_lightr1_step500_sp97.5.pt"
RANDOM_MASK="$MASK_DIR/random_baseline_lightr1_sp97.5_seed42.pt"

for f in "$ORACLE_MASK" "$RANDOM_MASK"; do
    if [ ! -f "$f" ]; then
        echo "[error] mask not found: $f"
        exit 1
    fi
done

mkdir -p "$OUT_DIR"
MASKS_JSON="${OUT_DIR}/masks.json"
cat > "$MASKS_JSON" <<JSON
[
  {"label": "Oracle-lightr1-step500", "path": "${ORACLE_MASK}"},
  {"label": "Random-sp97.5-seed42",   "path": "${RANDOM_MASK}"}
]
JSON

echo "MODEL      = $MODEL"
echo "OUT_DIR    = $OUT_DIR"
echo "MASKS_JSON = $MASKS_JSON"
echo ""

python src/analysis/probe_pair_masks.py \
    --model "$MODEL" \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUT_DIR" \
    --layer_stride 4 \
    --batch_size 8 \
    --max_length 256 \
    --cv_folds 5 \
    --pairs_per_pos 2 \
    --n_jobs 8 \
    --holdout_frac 0.2 \
    --use_holdout_as_test \
    --probe_C 0.01

echo ""
echo "Done: $(date)"
echo "Results: $OUT_DIR/"
echo "  3 configs: Baseline (no mask) / Oracle-lightr1-step500 / Random-sp97.5-seed42"

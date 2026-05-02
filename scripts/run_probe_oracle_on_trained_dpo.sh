#!/bin/bash
#SBATCH --job-name=probe_tulu3_oracle_dpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/probe_tulu3_oracle_dpo_%j.out
#SBATCH --error=logs/probe_tulu3_oracle_dpo_%j.err

# Apply the user's tulu3 oracle masks (step150, step500) to the public
# allenai/Llama-3.1-Tulu-3-8B-DPO checkpoint and run linear probes against
# the unmasked dense baseline. Both Allen's DPO model and the masks were
# derived on the Tulu3 preference mix, so the trajectory is data-matched
# (though hparam/seed-mismatched).

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

MODEL="${MODEL:-allenai/Llama-3.1-Tulu-3-8B-DPO}"
OUT_DIR="${OUT_DIR:-/scratch/xie.yiyi/probe_tulu3_oracle_on_allen_dpo}"
MASK_DIR="${MASK_DIR:-/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b}"
MASK_STEP150="$MASK_DIR/oracle_dpo_tulu3_step150_sp97.5.pt"
MASK_STEP500="$MASK_DIR/oracle_dpo_tulu3_step500_sp97.5.pt"

for f in "$MASK_STEP150" "$MASK_STEP500"; do
    if [ ! -f "$f" ]; then
        echo "[error] mask not found: $f"
        exit 1
    fi
done

mkdir -p "$OUT_DIR"
MASKS_JSON="${OUT_DIR}/tulu3_oracle_masks.json"
cat > "$MASKS_JSON" <<JSON
[
  {"label": "Oracle-DPO-tulu3-step150", "path": "${MASK_STEP150}"},
  {"label": "Oracle-DPO-tulu3-step500", "path": "${MASK_STEP500}"}
]
JSON

LAYER_STRIDE="${LAYER_STRIDE:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CV_FOLDS="${CV_FOLDS:-5}"
PAIRS_PER_POS="${PAIRS_PER_POS:-2}"
N_JOBS="${N_JOBS:-8}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"

echo ""
echo "========================================"
echo "MODEL       = $MODEL"
echo "MASKS_JSON  = $MASKS_JSON"
echo "OUT_DIR     = $OUT_DIR"
echo "========================================"

"$PYTHON_BIN" src/analysis/probe_pair_masks.py \
    --model "$MODEL" \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUT_DIR" \
    --layer_stride "$LAYER_STRIDE" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --cv_folds "$CV_FOLDS" \
    --pairs_per_pos "$PAIRS_PER_POS" \
    --n_jobs "$N_JOBS" \
    --holdout_frac "$HOLDOUT_FRAC" \
    --use_holdout_as_test

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results: $OUT_DIR/"
echo "  - probe_pair_results.json   (3 configs: baseline + step150 + step500)"
echo "  - probe_pair_heatmap_all.png"
echo "  - probe_pair_delta_all.png  (mask − baseline)"
echo "========================================"

#!/bin/bash
#SBATCH --job-name=probe_tulu3_subnetwork
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/probe_tulu3_subnetwork_%j.out
#SBATCH --error=logs/probe_tulu3_subnetwork_%j.err

# Probe the tulu3 oracle "subnetwork" against dense vs sparse-trained
# tulu3 DPO models, all on the user's own tulu3 trajectory (matched).
#   Run 1: dense tulu3 DPO + tulu3 oracle mask (post-hoc subnetwork)
#   Run 2: sparse-trained tulu3 DPO + tulu3 oracle mask (post-hoc subnetwork)
# Each run also produces a baseline-no-mask config for reference.

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

OUT_BASE="/scratch/xie.yiyi/probe_tulu3_subnetwork"
MASK="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_tulu3_step500_sp97.5.pt"
DENSE_MODEL="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
SPARSE_MODEL="/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_oracle_dpo_tulu3/sparse_dpo_tulu3_oracle_dpo_tulu3_500steps/final_model"

if [ ! -f "$MASK" ]; then echo "[error] mask missing"; exit 1; fi
if [ ! -d "$DENSE_MODEL" ]; then echo "[error] dense model missing"; exit 1; fi
if [ ! -d "$SPARSE_MODEL" ]; then echo "[error] sparse model missing"; exit 1; fi

mkdir -p "$OUT_BASE/dense" "$OUT_BASE/sparse"

cat > "$OUT_BASE/dense/masks.json" <<JSON
[{"label": "Oracle-DPO-tulu3-step500", "path": "${MASK}"}]
JSON
cp "$OUT_BASE/dense/masks.json" "$OUT_BASE/sparse/masks.json"

run_probe() {
    local tag="$1"
    local model="$2"
    local out_dir="${OUT_BASE}/${tag}"
    echo ""
    echo "========================================"
    echo "[${tag}] MODEL = $model"
    echo "[${tag}] OUT   = $out_dir"
    echo "========================================"
    python src/analysis/probe_pair_masks.py \
        --model "$model" \
        --masks_json "$out_dir/masks.json" \
        --output_dir "$out_dir" \
        --layer_stride 4 \
        --batch_size 8 \
        --max_length 256 \
        --cv_folds 5 \
        --pairs_per_pos 2 \
        --n_jobs 8 \
        --holdout_frac 0.2 \
        --use_holdout_as_test \
        --probe_C 0.01
}

run_probe "dense"  "$DENSE_MODEL"
run_probe "sparse" "$SPARSE_MODEL"

echo ""
echo "Done: $(date)"
echo "Results: $OUT_BASE/{dense,sparse}/probe_pair_{results.json,heatmap_all.png,delta_all.png}"

#!/bin/bash
#SBATCH --job-name=probe_subnet_grpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:30:00
#SBATCH --output=logs/probe_subnet_grpo_%j.out
#SBATCH --error=logs/probe_subnet_grpo_%j.err

# GRPO sub-network probe — scheme B (3x3 cross matrix).
# For each (ckpt × mask) pair, zero out mask=0 weights and probe the
# resulting sub-network. Answers: "on the same Oracle/CAV/Random subspace,
# which training (dense vs sparse-oracle) packed more useful info?"

set -euo pipefail

CUDA_LIB=$(dirname "$(find /usr /opt /lib64 /lib -name libcuda.so.1 2>/dev/null | head -n1)")
if [ -n "$CUDA_LIB" ] && [ "$CUDA_LIB" != "." ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
if [ -d /scratch/xie.yiyi/conda_envs/rl_casino ]; then
    conda activate /scratch/xie.yiyi/conda_envs/rl_casino
else
    conda activate /home/xie.yiyi/.conda/envs/rl_casino
fi

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/scratch/xie.yiyi/hf_cache
export HF_DATASETS_CACHE=/scratch/xie.yiyi/hf_cache/datasets

OUTPUT_DIR=/scratch/xie.yiyi/probe_subnet_grpo
mkdir -p "$OUTPUT_DIR"

# Mask paths (GRPO side)
ORACLE_MASK=/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt
CAV_MASK=/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_grpo.pt
RANDOM_MASK=/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/random_baseline_grpo_sp97.5_seed42.pt

# Checkpoints
BASELINE=meta-llama/Llama-3.1-8B-Instruct
DENSE_GRPO=/scratch/xie.yiyi/rl_casino_outputs/llama8b_dpo_mask_grpo_dense_baseline_1gpu_20260408_044634/checkpoints/meta_llama_llama_3_1_8b_instruct_math_220k_grpo_dense/checkpoint-200
SPARSE_GRPO_ORACLE=/scratch/xie.yiyi/rl_casino_outputs/llama8b_dpo_mask_grpo_sparse_magnitude_1gpu_20260409_031402/llama8b_dpo_mask_grpo_sparse_magnitude_1gpu_20260409_031402/final_model

# Group consecutive entries by ckpt path so probe_checkpoints.py preloads each
# ckpt exactly once and runs 3 mask probes with the same model.
CKPTS_JSON=$OUTPUT_DIR/ckpts.json
cat > "$CKPTS_JSON" <<JSON
[
  {"label": "Baseline ∩ Oracle",          "path": "$BASELINE",           "mask_path": "$ORACLE_MASK"},
  {"label": "Baseline ∩ CAV",             "path": "$BASELINE",           "mask_path": "$CAV_MASK"},
  {"label": "Baseline ∩ Random",          "path": "$BASELINE",           "mask_path": "$RANDOM_MASK"},
  {"label": "Dense-GRPO ∩ Oracle",        "path": "$DENSE_GRPO",         "mask_path": "$ORACLE_MASK"},
  {"label": "Dense-GRPO ∩ CAV",           "path": "$DENSE_GRPO",         "mask_path": "$CAV_MASK"},
  {"label": "Dense-GRPO ∩ Random",        "path": "$DENSE_GRPO",         "mask_path": "$RANDOM_MASK"},
  {"label": "Sparse-GRPO-Oracle ∩ Oracle", "path": "$SPARSE_GRPO_ORACLE", "mask_path": "$ORACLE_MASK"}
]
JSON
echo "Wrote ckpts JSON: $CKPTS_JSON ($(python3 -c "import json; print(len(json.load(open('$CKPTS_JSON'))))") configs)"

python src/analysis/probe_checkpoints.py \
    --ckpts_json "$CKPTS_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --skip_baseline \
    --layer_stride 4 \
    --batch_size 8 \
    --max_length 384 \
    --cv_folds 5 \
    --pairs_per_pos 2 \
    --n_jobs 8 \
    --probe_C 0.1 \
    --preference_samples_per_class 1500 \
    --holdout_frac 0.4 \
    --use_holdout_as_test

echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=multi_mask_jcka
#SBATCH --output=logs/multi_mask_jcka_%j.out
#SBATCH --error=logs/multi_mask_jcka_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate /home/xie.yiyi/.conda/envs/rl_casino

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"

ORACLE_DIR=/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b
OUT_DIR=/scratch/xie.yiyi/transfer_v1/multi_mask_analysis
mkdir -p "$OUT_DIR"

MASKS_JSON="$OUT_DIR/masks.json"
cat > "$MASKS_JSON" << JSONEOF
[
  {"label": "DPO-LightR1-500",  "path": "$ORACLE_DIR/oracle_dpo_lightr1_step500_sp97.5.pt"},
  {"label": "DPO-Tulu3-500",    "path": "$ORACLE_DIR/oracle_dpo_tulu3_step500_sp97.5.pt"},
  {"label": "GRPO-OpenR1-500",  "path": "$ORACLE_DIR/oracle_grpo_math220k_step500_sp97.5.pt"},
  {"label": "Random",           "path": "$ORACLE_DIR/random_baseline_lightr1_sp97.5_seed42.pt"}
]
JSONEOF

echo "=========================================="
echo "Multi-mask Jaccard + CKA (mask vs mask)"
echo "Started: $(date)"
echo "=========================================="
cat "$MASKS_JSON"
echo ""
echo "=========================================="

python scripts/multi_mask_jaccard_cka.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUT_DIR" \
    --layer_substr down_proj \
    --n_samples 64 \
    --max_length 512 \
    --batch_size 4 \
    --seed 42 \
    --calibration_dataset tulu3

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -lh "$OUT_DIR" 2>&1
echo "=========================================="

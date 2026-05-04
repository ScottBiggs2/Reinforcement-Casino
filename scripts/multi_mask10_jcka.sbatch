#!/bin/bash
#SBATCH --job-name=mm10_jcka
#SBATCH --output=logs/mm10_jcka_%j.out
#SBATCH --error=logs/mm10_jcka_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
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
WARM_DIR=/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b
OUT_DIR=/scratch/xie.yiyi/transfer_v1/multi_mask10_analysis
mkdir -p "$OUT_DIR"

MASKS_JSON="$OUT_DIR/masks.json"
cat > "$MASKS_JSON" << 'JSONEOF'
[
  {
    "label":    "DPO-LightR1-s150-sp97.5",
    "family":   "DPO-oracle",
    "dataset":  "LightR1",
    "step":     150,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step150_sp97.5.pt"
  },
  {
    "label":    "DPO-LightR1-s500-sp97.5",
    "family":   "DPO-oracle",
    "dataset":  "LightR1",
    "step":     500,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step500_sp97.5.pt"
  },
  {
    "label":    "DPO-Tulu3-s150-sp97.5",
    "family":   "DPO-oracle",
    "dataset":  "Tulu3",
    "step":     150,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_tulu3_step150_sp97.5.pt"
  },
  {
    "label":    "DPO-Tulu3-s500-sp97.5",
    "family":   "DPO-oracle",
    "dataset":  "Tulu3",
    "step":     500,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_tulu3_step500_sp97.5.pt"
  },
  {
    "label":    "GRPO-Math220k-s500-sp97.5",
    "family":   "GRPO-oracle",
    "dataset":  "OpenR1-Math220k",
    "step":     500,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt"
  },
  {
    "label":    "GRPO-Math220k-s850-sp97.5",
    "family":   "GRPO-oracle",
    "dataset":  "OpenR1-Math220k",
    "step":     850,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step850_sp97.5.pt"
  },
  {
    "label":    "Random-LightR1-sp97.5-s42",
    "family":   "Random",
    "dataset":  "LightR1",
    "step":     null,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/random_baseline_lightr1_sp97.5_seed42.pt"
  },
  {
    "label":    "Random-Tulu3-sp97.5-s42",
    "family":   "Random",
    "dataset":  "Tulu3",
    "step":     null,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/random_baseline_tulu3_sp97.5_seed42.pt"
  },
  {
    "label":    "WarmMag-LightR1-s200-sp97.5",
    "family":   "WarmMag",
    "dataset":  "LightR1",
    "step":     200,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b/warm_magnitude_dpo_lightr1_step200_sp97.5.pt"
  },
  {
    "label":    "WarmMag-Tulu3-s200-sp97.5",
    "family":   "WarmMag",
    "dataset":  "Tulu3",
    "step":     200,
    "sparsity": 97.5,
    "path":     "/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b/warm_magnitude_dpo_tulu3_step200_sp97.5.pt"
  }
]
JSONEOF

echo "=========================================="
echo "Multi-mask (10) Jaccard + CKA"
echo "Started: $(date)"
echo "=========================================="
cat "$MASKS_JSON"
echo ""
echo "=========================================="

# 10 masks → C(10,2)=45 pairs.
# Pre-flight: every mask path must resolve.
python - << 'PY'
import json, os, sys
m = json.load(open(os.environ.get("MASKS_JSON_PATH","/scratch/xie.yiyi/transfer_v1/multi_mask10_analysis/masks.json")))
missing = [e["path"] for e in m if not os.path.isfile(e["path"])]
if missing:
    print("MISSING MASKS:")
    for p in missing: print("  -", p)
    sys.exit(2)
print(f"All {len(m)} mask files present.")
PY
export MASKS_JSON_PATH="$MASKS_JSON"

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

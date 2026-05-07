#!/bin/bash
#SBATCH --job-name=dpo_grpo_deep
#SBATCH --output=logs/dpo_grpo_deep_%j.out
#SBATCH --error=logs/dpo_grpo_deep_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
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
OUT_DIR=/scratch/xie.yiyi/transfer_v1/dpo_grpo_deep
mkdir -p "$OUT_DIR"

MASKS_JSON="$OUT_DIR/masks.json"
cat > "$MASKS_JSON" << 'JSONEOF'
[
  {"label":"DPO-LightR1-s500", "path":"/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step500_sp97.5.pt"},
  {"label":"DPO-LightR1-s150", "path":"/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step150_sp97.5.pt"},
  {"label":"DPO-Tulu3-s500",   "path":"/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_tulu3_step500_sp97.5.pt"},
  {"label":"DPO-Tulu3-s150",   "path":"/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_tulu3_step150_sp97.5.pt"},
  {"label":"GRPO-Math220k-s500","path":"/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt"},
  {"label":"GRPO-Math220k-s850","path":"/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step850_sp97.5.pt"},
  {"label":"Random-LightR1",   "path":"/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/random_baseline_lightr1_sp97.5_seed42.pt"}
]
JSONEOF

echo "=========================================="
echo "DPO vs GRPO deep mask comparison"
echo "Started: $(date)"
echo "Output : $OUT_DIR"
echo "=========================================="
cat "$MASKS_JSON"
echo ""
echo "=========================================="

python - << 'PY'
import json, os, sys
m = json.load(open("/scratch/xie.yiyi/transfer_v1/dpo_grpo_deep/masks.json"))
missing = [e["path"] for e in m if not os.path.isfile(e["path"])]
if missing:
    print("MISSING MASKS:")
    for p in missing: print("  -", p)
    sys.exit(2)
print(f"All {len(m)} mask files present.")
PY

# 2-stage run: stage 1 = no magnitude (fast static analysis only).
# Stage 2 = base-model magnitude histogram on selected DPO/GRPO step-500 pair.
python scripts/mask_dpo_grpo_deep.py \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUT_DIR" \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --magnitude_pairs "DPO-LightR1-s500::GRPO-Math220k-s500,DPO-Tulu3-s500::GRPO-Math220k-s500" \
    --skip_mlp_neuron

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -lh "$OUT_DIR" 2>&1
echo "=========================================="

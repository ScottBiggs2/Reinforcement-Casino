#!/bin/bash
#SBATCH --job-name=regen_cav_masks
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/regen_cav_masks_%j.out
#SBATCH --error=logs/regen_cav_masks_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started  : $(date)"
echo "========================================"

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PRIMARY="/scratch/xie.yiyi/conda_envs/rl_casino"
CONDA_ENV_FALLBACK="/home/xie.yiyi/.conda/envs/rl_casino"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  if [ -d "$CONDA_ENV_PRIMARY" ]; then
    conda activate "$CONDA_ENV_PRIMARY"
  elif [ -d "$CONDA_ENV_FALLBACK" ]; then
    conda activate "$CONDA_ENV_FALLBACK"
  fi
fi

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export HF_DATASETS_CACHE="/scratch/xie.yiyi/hf_cache/datasets"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SPARSITY="${SPARSITY:-97.5}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
DPO_TARGET_STEP="${DPO_TARGET_STEP:-50}"
GRPO_TARGET_STEP="${GRPO_TARGET_STEP:-200}"

COLD_DIR="/scratch/xie.yiyi/rl_casino_masks/llama8b_cold"
WARM_DPO_DIR="/scratch/xie.yiyi/rl_casino_masks/llama8b"
WARM_GRPO_DIR="/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo"
mkdir -p "$COLD_DIR" "$WARM_DPO_DIR" "$WARM_GRPO_DIR"

DPO_CAV="${COLD_DIR}/cold_cav_dpo.pt"
GRPO_CAV="${COLD_DIR}/cold_cav_grpo.pt"
DPO_ORACLE="${WARM_DPO_DIR}/warm_magnitude_step50_sp97.5.pt"
GRPO_ORACLE="${WARM_GRPO_DIR}/warm_magnitude_grpo.pt"
DPO_RANDOM="${WARM_DPO_DIR}/random_baseline_dpo_sp97.5_seed${RANDOM_SEED}.pt"
GRPO_RANDOM="${WARM_GRPO_DIR}/random_baseline_grpo_sp97.5_seed${RANDOM_SEED}.pt"

find_latest_delta_file() {
  local step="$1"
  local include_pat="$2"
  local exclude_pat="$3"
  local candidates
  candidates=$(find /scratch/xie.yiyi -type f -name "deltas_step_${step}.pt" 2>/dev/null | grep -E "$include_pat" || true)
  if [ -n "$exclude_pat" ]; then
    candidates=$(echo "$candidates" | grep -Ev "$exclude_pat" || true)
  fi
  if [ -z "${candidates// }" ]; then
    echo ""
    return 0
  fi
  local latest
  latest=$(while IFS= read -r f; do
    [ -n "$f" ] || continue
    stat -c "%Y %n" "$f"
  done <<< "$candidates" | sort -nr | head -n1 | cut -d" " -f2-)
  echo "$latest"
}

DPO_DELTA_FILE="${DPO_DELTA_FILE:-}"
GRPO_DELTA_FILE="${GRPO_DELTA_FILE:-}"

if [ -z "$DPO_DELTA_FILE" ]; then
  DPO_DELTA_FILE=$(find_latest_delta_file "$DPO_TARGET_STEP" "rl_casino_(outputs|train)|dpo|delta" "grpo")
fi
if [ -z "$GRPO_DELTA_FILE" ]; then
  GRPO_DELTA_FILE=$(find_latest_delta_file "$GRPO_TARGET_STEP" "rl_casino_(outputs|train)|grpo|delta" "")
fi

if [ -z "$DPO_DELTA_FILE" ] || [ ! -f "$DPO_DELTA_FILE" ]; then
  echo "FATAL: could not locate DPO delta file for step ${DPO_TARGET_STEP}."
  echo "Set DPO_DELTA_FILE explicitly when submitting this job."
  exit 1
fi
if [ -z "$GRPO_DELTA_FILE" ] || [ ! -f "$GRPO_DELTA_FILE" ]; then
  echo "FATAL: could not locate GRPO delta file for step ${GRPO_TARGET_STEP}."
  echo "Set GRPO_DELTA_FILE explicitly when submitting this job."
  exit 1
fi

DPO_DELTA_DIR=$(dirname "$DPO_DELTA_FILE")
GRPO_DELTA_DIR=$(dirname "$GRPO_DELTA_FILE")

echo "[config] MODEL=$MODEL"
echo "[config] SPARSITY=$SPARSITY"
echo "[config] DPO_DELTA_FILE=$DPO_DELTA_FILE"
echo "[config] GRPO_DELTA_FILE=$GRPO_DELTA_FILE"

echo "[1/4] Regenerating Oracle masks (warm magnitude)"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$DPO_DELTA_DIR" \
  --method magnitude \
  --sparsity_percent "$SPARSITY" \
  --target_step "$DPO_TARGET_STEP" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard \
  --output_file "$DPO_ORACLE"

python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$GRPO_DELTA_DIR" \
  --method magnitude \
  --sparsity_percent "$SPARSITY" \
  --target_step "$GRPO_TARGET_STEP" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard \
  --output_file "$GRPO_ORACLE"

echo "[2/4] Regenerating Cold-CAV masks"
INFER_HELP="$(python src/cold_start/inference_mask_finder.py --help 2>&1 || true)"
CAV_EXTRA_ARGS=()
if echo "$INFER_HELP" | grep -q -- "--min_layer_keep_ratio"; then
  CAV_EXTRA_ARGS+=(--min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO")
  echo "[cav] using --min_layer_keep_ratio=$MIN_LAYER_KEEP_RATIO"
else
  echo "[cav] --min_layer_keep_ratio not supported by this inference_mask_finder.py; proceeding without it"
fi

python src/cold_start/inference_mask_finder.py \
  --model_name "$MODEL" \
  --method cav \
  --mode dpo \
  --n_samples 256 \
  --sparsity "$SPARSITY" \
  --batch_size 8 \
  --max_length 256 \
  "${CAV_EXTRA_ARGS[@]}" \
  --verbose \
  --output "$DPO_CAV"

python src/cold_start/inference_mask_finder.py \
  --model_name "$MODEL" \
  --method cav \
  --mode grpo \
  --n_samples 256 \
  --sparsity "$SPARSITY" \
  --batch_size 8 \
  --max_length 256 \
  "${CAV_EXTRA_ARGS[@]}" \
  --verbose \
  --output "$GRPO_CAV"

echo "[3/4] Regenerating Random baseline masks"
python src/warm_start/random_mask_baseline.py \
  --reference_mask "$DPO_ORACLE" \
  --sparsity_percent "$SPARSITY" \
  --output_file "$DPO_RANDOM" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --seed "$RANDOM_SEED" \
  --compare_to_reference

python src/warm_start/random_mask_baseline.py \
  --reference_mask "$GRPO_ORACLE" \
  --sparsity_percent "$SPARSITY" \
  --output_file "$GRPO_RANDOM" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --seed "$RANDOM_SEED" \
  --compare_to_reference

echo "[4/4] Verifying mask quality and consistency"
VALIDATION_JSON="/scratch/xie.yiyi/rl_casino_outputs/mask_validation_cav_random_oracle_$(date +%Y%m%d_%H%M%S).json"
python - "$DPO_CAV" "$DPO_ORACLE" "$DPO_RANDOM" "$GRPO_CAV" "$GRPO_ORACLE" "$GRPO_RANDOM" "$VALIDATION_JSON" <<'PYEOF'
import json
import sys
import torch
from src.utils.mask_utils import compute_jaccard_similarity

paths = {
    "dpo_cav": sys.argv[1],
    "dpo_oracle": sys.argv[2],
    "dpo_random": sys.argv[3],
    "grpo_cav": sys.argv[4],
    "grpo_oracle": sys.argv[5],
    "grpo_random": sys.argv[6],
}
out_json = sys.argv[7]

def load_masks(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "masks" in obj:
        return obj["masks"], obj.get("metadata", {})
    return obj, {}

def sparsity(mask_dict):
    kept = 0
    total = 0
    for t in mask_dict.values():
        x = t.float()
        kept += float(x.sum().item())
        total += x.numel()
    density = kept / total
    return 100.0 * (1.0 - density)

m = {}
meta = {}
for k, p in paths.items():
    masks, md = load_masks(p)
    m[k] = masks
    meta[k] = md

res = {
    "paths": paths,
    "sparsity_percent": {k: sparsity(v) for k, v in m.items()},
    "jaccard": {
        "dpo_cav_vs_oracle": compute_jaccard_similarity(m["dpo_cav"], m["dpo_oracle"]),
        "dpo_random_vs_oracle": compute_jaccard_similarity(m["dpo_random"], m["dpo_oracle"]),
        "grpo_cav_vs_oracle": compute_jaccard_similarity(m["grpo_cav"], m["grpo_oracle"]),
        "grpo_random_vs_oracle": compute_jaccard_similarity(m["grpo_random"], m["grpo_oracle"]),
    },
    "metadata": meta,
}

res["checks"] = {
    "dpo_cav_beats_random": res["jaccard"]["dpo_cav_vs_oracle"]["aggregate_jaccard"] > res["jaccard"]["dpo_random_vs_oracle"]["aggregate_jaccard"],
    "grpo_cav_beats_random": res["jaccard"]["grpo_cav_vs_oracle"]["aggregate_jaccard"] > res["jaccard"]["grpo_random_vs_oracle"]["aggregate_jaccard"],
}

with open(out_json, "w") as f:
    json.dump(res, f, indent=2)

print("=== Validation Summary ===")
print(json.dumps({
    "sparsity_percent": res["sparsity_percent"],
    "dpo_jaccard": {
      "cav_vs_oracle": res["jaccard"]["dpo_cav_vs_oracle"]["aggregate_jaccard"],
      "random_vs_oracle": res["jaccard"]["dpo_random_vs_oracle"]["aggregate_jaccard"],
    },
    "grpo_jaccard": {
      "cav_vs_oracle": res["jaccard"]["grpo_cav_vs_oracle"]["aggregate_jaccard"],
      "random_vs_oracle": res["jaccard"]["grpo_random_vs_oracle"]["aggregate_jaccard"],
    },
    "checks": res["checks"],
    "validation_json": out_json,
}, indent=2))
PYEOF

echo "========================================"
echo "Done: $(date)"
echo "DPO_CAV:    $DPO_CAV"
echo "GRPO_CAV:   $GRPO_CAV"
echo "DPO_ORACLE: $DPO_ORACLE"
echo "GRPO_ORACLE:$GRPO_ORACLE"
echo "DPO_RANDOM: $DPO_RANDOM"
echo "GRPO_RANDOM:$GRPO_RANDOM"
echo "========================================"

#!/usr/bin/env bash
# Build (1) warm magnitude step200 mask, (2) checkpoint-diff oracle mask, (3) matched random baseline
# from an existing dense DPO run (with deltas + checkpoint directories).
#
# This is intended for creating the Tulu3 (or other DPO dataset key) masks you want to compare later.
#
# Drop-in (Explorer, repo root):
#   export HF_TOKEN="${HF_TOKEN:?}"   # for gated Llama weights (oracle initial_model)
#   sbatch scripts/sbatch_make_mag_oracle_random_masks_from_run.sh
#
# Overrides (recommended to pin the dense run you want):
#   export PIPELINE_RUN_ID=<dense_run_id_with_deltas_and_checkpoints>
#   export DPO_DATASET_KEY=tulu3
#   export MODEL=meta-llama/Llama-3.1-8B-Instruct
#   export CHECKPOINT_STEP=500
#   export TARGET_STEP_DPO=200
#   export SPARSITY_PERCENT=97.5
#   export MASK_RUN_ID=manual_masks_<tag>   # output folder under /scratch/$USER/rl_casino_masks/
#
# Outputs:
#   /scratch/$USER/rl_casino_masks/$MASK_RUN_ID/
#     warm_magnitude_<model>_<ds>_sparsity97.5pct_step200.pt
#     checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct.pt
#     random_baseline_<model>_<ds>_sparsity97.5pct_step200_seed42.pt
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=mk_masks_mag_oracle_rand
#SBATCH --output=logs/mk_masks_mag_oracle_rand_%j.out
#SBATCH --error=logs/mk_masks_mag_oracle_rand_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-${RUN_ID_OVERRIDE:-dpo500_sparse_lr1_grasp_elem_base_lr1_grasp_elem_base_500_20260503_164435}}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DPO_DATASET_KEY="${DPO_DATASET_KEY:-tulu3}"

SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
TARGET_STEP_DPO="${TARGET_STEP_DPO:-200}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-500}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
RANDOM_MASK_SEED="${RANDOM_MASK_SEED:-42}"

MASK_RUN_ID="${MASK_RUN_ID:-manual_masks_${SLURM_JOB_ID:-local}}"
OUT_DIR="${MASK_OUT_BASE}/${MASK_RUN_ID}"
mkdir -p "$OUT_DIR"

echo "=== Build masks from dense run ==="
echo "PIPELINE_RUN_ID=${PIPELINE_RUN_ID}"
echo "MODEL=${MODEL}"
echo "DPO_DATASET_KEY=${DPO_DATASET_KEY}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT}  TARGET_STEP_DPO=${TARGET_STEP_DPO}  CHECKPOINT_STEP=${CHECKPOINT_STEP}"
echo "OUT_DIR=${OUT_DIR}"

# Resolve (model_sanitized, ds_sanitized, subdir) in the same way as pipeline_common.sh
mapfile -t _p < <(PIPELINE_MODEL="$MODEL" PIPELINE_DS_KEY="$DPO_DATASET_KEY" PIPELINE_REPO="$REPO_ROOT" "$TRAIN_PY" -c "
import os, sys
sys.path.insert(0, os.environ['PIPELINE_REPO'])
from src.utils.dataset_registry import get_dataset_config
def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace('/', '_').replace('-', '_').lower()
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized.strip('_')
m = os.environ['PIPELINE_MODEL']
k = os.environ['PIPELINE_DS_KEY']
cfg = get_dataset_config(k)
ms = sanitize_model_name(m)
ds = cfg['sanitized_name']
print(ms)
print(ds)
print(f'{ms}_{ds}')
")
MODEL_SAN="${_p[0]}"
DS_SAN="${_p[1]}"
SUBDIR="${_p[2]}"

DELTA_DIR="${TRAIN_OUT_BASE}/${PIPELINE_RUN_ID}/deltas/${SUBDIR}"
CKPT_DIR="${TRAIN_OUT_BASE}/${PIPELINE_RUN_ID}/checkpoints/${SUBDIR}/checkpoint-${CHECKPOINT_STEP}"

if [ ! -d "$DELTA_DIR" ]; then
  echo "ERROR: deltas dir not found: $DELTA_DIR" >&2
  echo "  Fix: export PIPELINE_RUN_ID=<dense run id that wrote deltas/> and DPO_DATASET_KEY=<dataset key used>." >&2
  exit 2
fi
if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: checkpoint dir not found: $CKPT_DIR" >&2
  echo "  Fix: export CHECKPOINT_STEP=<step> or PIPELINE_RUN_ID/DPO_DATASET_KEY to match your dense run." >&2
  exit 2
fi

MAG_OUT="${OUT_DIR}/warm_magnitude_${MODEL_SAN}_${DS_SAN}_sparsity${SPARSITY_PERCENT}pct_step${TARGET_STEP_DPO}.pt"
ORACLE_OUT="${OUT_DIR}/checkpoint_diff_ground_truth_checkpoint-${CHECKPOINT_STEP}_sparsity${SPARSITY_PERCENT}pct.pt"
RAND_OUT="${OUT_DIR}/random_baseline_${MODEL_SAN}_${DS_SAN}_sparsity${SPARSITY_PERCENT}pct_step${TARGET_STEP_DPO}_seed${RANDOM_MASK_SEED}.pt"

echo ""
echo "--- (1) Warm magnitude mask from deltas ---"
echo "DELTA_DIR=$DELTA_DIR"
echo "OUT=$MAG_OUT"
"$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$DELTA_DIR" \
  --method magnitude \
  --sparsity_percent "$SPARSITY_PERCENT" \
  --target_step "$TARGET_STEP_DPO" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "$MAG_OUT"

echo ""
echo "--- (2) Oracle checkpoint-diff ground truth mask ---"
echo "CKPT_DIR=$CKPT_DIR"
echo "OUT=$ORACLE_OUT"
"$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
  --initial_model "$MODEL" \
  --final_model "$CKPT_DIR" \
  --sparsity_percent "$SPARSITY_PERCENT" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "$ORACLE_OUT"

echo ""
echo "--- (3) Random baseline mask (matched to magnitude topology) ---"
echo "REF=$MAG_OUT"
echo "OUT=$RAND_OUT"
"$TRAIN_PY" src/warm_start/random_mask_baseline.py \
  --reference_mask "$MAG_OUT" \
  --sparsity_percent "$SPARSITY_PERCENT" \
  --seed "$RANDOM_MASK_SEED" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "$RAND_OUT"

echo ""
echo "✓ Done. Masks written under: $OUT_DIR"
ls -lh "$OUT_DIR" | sed -n '1,200p'


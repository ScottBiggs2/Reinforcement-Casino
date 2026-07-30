#!/bin/bash
# Oracle (checkpoint-diff) mask sweep over sparsity for Llama-3.1-8B-Instruct.
#
# Usage:
#   DATASET=tulu3 SRC_CKPT=/scratch/.../checkpoint-500 \
#     REF_MASK=/scratch/.../oracle_dpo_tulu3_step500_sp97.5.pt \
#     sbatch scripts/transfer_p2_oracle_sweep_llama8b.sbatch
#
# rho is the fraction of weights ZEROED. Array index -> rho:
#   0 -> 80    (keep 20%,  ~1.6B params)
#   1 -> 90    (keep 10%,  ~0.80B)   measured update-mass saturation knee
#   2 -> 97.5  (keep 2.5%, ~0.20B)   the level the paper reports -- anchor
#   3 -> 99    (keep 1%,   ~0.08B)   closest to LoRA r=16 scale (~42M)
#
# min_layer_keep_ratio is held at 0.0025 across all four levels on purpose: rho is
# the only knob. Note the floor is a larger share of the budget as rho rises
# (10% of it at rho=97.5, 25% at rho=99); that is recorded, not corrected.
#
# CPU partition, no GPU. checkpoint_diff_mask_finder is pure CPU tensor work once
# --force_cpu is set, and the chunked selector already runs on CPU at 8B scale.
# Requesting no GPU sidesteps both the h200 queue (one node in multigpu) and the
# RC 15-minute idle-GPU auto-cancel.
#SBATCH --job-name=p2_oracle_sweep
#SBATCH --output=logs/p2_oracle_sweep_%A_%a.out
#SBATCH --error=logs/p2_oracle_sweep_%A_%a.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-3

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

: "${DATASET:?Set DATASET=tulu3 | lightr1}"
: "${SRC_CKPT:?Set SRC_CKPT=/scratch/.../checkpoint-500}"
: "${OUT_TAG:?Set OUT_TAG=<short id of the source dense run>, e.g. deltalog}"
REF_MASK="${REF_MASK:-}"

# OUT_TAG is mandatory because the un-tagged name oracle_dpo_<ds>_step500_sp97.5.pt
# is already taken by the mask from the (now deleted) original dense run. Without a
# tag the rho=97.5 task collides with it, skips generation, and the gate below ends
# up comparing that file against itself and reporting Jaccard = 1.0000.

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
MIN_LAYER_KEEP_RATIO=0.0025
OUT_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
mkdir -p "$OUT_DIR"

SPARSITIES=(80 90 97.5 99)
SP="${SPARSITIES[$SLURM_ARRAY_TASK_ID]}"

if [ ! -d "$SRC_CKPT" ]; then echo "FATAL: SRC_CKPT not found: $SRC_CKPT"; exit 1; fi
STEP=$(basename "$SRC_CKPT" | sed 's/checkpoint-//')
OUT="$OUT_DIR/oracle_dpo_${DATASET}_step${STEP}_sp${SP}_src-${OUT_TAG}.pt"

if [ -n "$REF_MASK" ] && [ "$REF_MASK" = "$OUT" ]; then
  echo "FATAL: REF_MASK equals OUT ($OUT). The gate would compare a file to itself."
  exit 1
fi

echo "=========================================="
echo "Phase 2 sweep: oracle mask | $DATASET | rho=${SP}%"
echo "Job    : ${SLURM_JOB_ID:-local} (array task $SLURM_ARRAY_TASK_ID)"
echo "Started: $(date)"
echo "=========================================="
echo "BASE  = $MODEL_BASE"
echo "SRC   = $SRC_CKPT  (step $STEP)"
echo "OUT   = $OUT"
echo "floor = $MIN_LAYER_KEEP_RATIO (held constant across the sweep)"
echo "=========================================="

if [ -f "$OUT" ]; then
  echo "Mask already exists, skipping generation: $OUT"
else
  python src/warm_start/checkpoint_diff_mask_finder.py \
      --initial_model "$MODEL_BASE" \
      --final_model "$SRC_CKPT" \
      --sparsity_percent "$SP" \
      --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
      --force_cpu \
      --output_file "$OUT"
fi

echo ""
echo "=== measured density (does the mask actually keep 1-rho?) ==="
python - "$OUT" <<'PY'
import sys, torch
obj = torch.load(sys.argv[1], map_location="cpu", mmap=True)
masks = obj.get("masks", obj)
kept = total = 0
for t in masks.values():
    kept += int(t.sum()); total += t.numel()
print(f"kept {kept:,} / {total:,} = {100*kept/total:.4f}%  (rho = {100-100*kept/total:.4f}%)")
print("metadata:", obj.get("metadata", {}))
PY

# Gate: only the rho=97.5 task compares against the surviving paper mask. Its source
# dense run was deleted, so this Jaccard is the only number that says whether the new
# sweep can be read alongside the published 97.5 point or has to stand on its own.
if [ "$SP" = "97.5" ] && [ -n "$REF_MASK" ] && [ -f "$REF_MASK" ]; then
  echo ""
  echo "=== GATE: Jaccard(new sp97.5, paper sp97.5) ==="
  echo "new  : $OUT"
  echo "paper: $REF_MASK"
  python - "$OUT" "$REF_MASK" <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", mmap=True)
b = torch.load(sys.argv[2], map_location="cpu", mmap=True)
ma, mb = a.get("masks", a), b.get("masks", b)
print("paper mask source:", b.get("metadata", {}).get("final_model"))
inter = union = 0
for k in set(ma) & set(mb):
    x, y = ma[k].bool(), mb[k].bool()
    inter += int((x & y).sum()); union += int((x | y).sum())
j = inter / max(union, 1)
print(f"Jaccard = {j:.4f}   (chance level at rho=97.5 is ~1.27e-2)")
print("high -> new sweep is comparable to the published point")
print("low  -> report the sweep self-contained; do not splice the paper number in")
PY
fi

echo ""
echo "Finished: $(date)"
ls -lh "$OUT"

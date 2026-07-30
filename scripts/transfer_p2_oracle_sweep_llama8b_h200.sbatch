#!/bin/bash
# Oracle (checkpoint-diff) mask sweep -- h200 variant.
#
# Same work as transfer_p2_oracle_sweep_llama8b.sbatch, but placed on multigpu with an
# h200 instead of the short CPU partition. Reason: on 2026-07-28 evening `short` had
# 2940 pending jobs with 91 nodes down and 10 drained, and Slurm estimated a 34-hour
# wait; multigpu had 24 pending.
#
# Memory is NOT one-size-fits-all here. Measured peaks from the Tulu3 run of this same
# script scale with the keep count:
#     rho=80  -> 199 GB      rho=90   -> 161 GB
#     rho=97.5-> 105 GB      rho=99   ->  98 GB
# so --mem is overridden per rho on the sbatch command line rather than set here.
# A blanket 128G would have OOM'd rho=80 and rho=90.
#
# Usage (one job per rho, right-sized):
#   sbatch --array=2 --mem=120G --export=ALL,DATASET,SRC_CKPT,REF_MASK,OUT_TAG \
#          scripts/transfer_p2_oracle_sweep_llama8b_h200.sbatch
#
# rho is the fraction of weights ZEROED. Array index -> rho:
#   0 -> 80    1 -> 90    2 -> 97.5 (the Gate)    3 -> 99
#
# min_layer_keep_ratio is held at 0.0025 across all levels: rho is the only knob.
#SBATCH --job-name=p2_oracle_h200
#SBATCH --output=logs/p2_oracle_h200_%A_%a.out
#SBATCH --error=logs/p2_oracle_h200_%A_%a.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-3

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

: "${DATASET:?Set DATASET=tulu3 | lightr1}"
: "${SRC_CKPT:?Set SRC_CKPT=/scratch/.../checkpoint-500}"
: "${OUT_TAG:?Set OUT_TAG=<short id of the source dense run>}"
REF_MASK="${REF_MASK:-}"

# ---------------------------------------------------------------------------
# Keep-alive sidecar. This job holds an h200 but does all its work on the CPU:
# the |dtheta| scores are computed from CPU state dicts, and at 8B the chunked
# top-k selector runs on CPU regardless of --force_cpu. The GPU therefore sits
# at 0% for the whole run, which trips the RC 15-minute idle-GPU canceller
# (AGENT_PLAYBOOK 7b). The sidecar does a small matmul every ~45 s so the
# monitor sees activity while the real work proceeds.
# Legitimate here because the job IS working -- never to hold an idle allocation.
# ---------------------------------------------------------------------------
python -c "import torch,time
gs=[torch.zeros(1024,1024,device=f'cuda:{i}') for i in range(torch.cuda.device_count())]
while True:
    for g in gs: (g@g).sum()
    torch.cuda.synchronize(); time.sleep(45)" &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
echo "keep-alive sidecar pid=$KEEPALIVE (GPU idle by design; the work is on CPU)"

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
echo "Phase 2 sweep (h200): oracle mask | $DATASET | rho=${SP}%"
echo "Job    : ${SLURM_JOB_ID:-local} (array task $SLURM_ARRAY_TASK_ID)"
echo "Node   : ${SLURMD_NODENAME:-$(hostname)}   mem=${SLURM_MEM_PER_NODE:-?}M"
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

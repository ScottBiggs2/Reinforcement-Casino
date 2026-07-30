#!/bin/bash
# Oracle (checkpoint-diff) mask sweep for Light-R1, all four rho in ONE allocation.
#
# Why packed instead of four jobs: `sbatch --test-only` on 2026-07-28 quoted ~7 h of
# queue wait for every shape tried, which dominates a 10-minute mask. Four jobs pay that
# wait four times and burn four of the eight multigpu submit slots (QOS multigpu:
# gres/gpu=8, MaxJobsPU 4, MaxSubmitPU 8). One job pays it once and costs one slot.
# Order is cheapest-first so the fast arms can start training while rho=80 is still
# selecting: 99 (0:10) -> 97.5 (0:25) -> 90 (1:31) -> 80 (2:58), ~5:04 total.
#
# Route: score |deltas_step_500| instead of diffing two full state dicts. The delta was
# written by FlexibleCheckpointCallback as `param - base_state` with base_state = theta(0),
# so |delta_500| == |theta(500) - theta(0)| exactly -- the same quantity
# checkpoint_diff_mask_finder.py computes. Both then call the same
# create_mask_from_scores_gpu_efficient with the same sparsity, min_layer_keep_ratio and
# pooling, and neither overrides add_tie_break_noise (True) or tie_break_noise_scale (1e-6).
# NOT even_better_mask_finder.py --method magnitude, which accumulates
# sum_k |theta(k) - theta(0)| over every logged step -- a different score that would
# silently change method between the Tulu3 and Light-R1 halves of the sweep.
#
# Sizing: --mem=200G. create_mask_from_scores_gpu_efficient clones every score tensor
# (src/utils/mask_utils.py:623), so a second full 32 GB copy is materialised inside the
# selector no matter how carefully the caller avoids one. Measured checkpoint-diff peaks
# were 199/161/105/98 GB for rho=80/90/97.5/99; a packed job must cover the rho=80 peak.
#
# Partition: bare --gres=gpu:1 on multigpu, NOT h200. This is CPU-only work and the
# v100-pcie nodes (c2205-c2207, 480 G) are the right home for it; feedback_must_use_h200
# exempts mask generation, and DO_NOT_REPEAT's "pin a100/h200" rule is about jobs doing
# 8B forward passes, which this does none of. `short` was measured at a 2026-08-03 start.
#
# Usage:
#   S=/scratch/$USER
#   sbatch --partition=multigpu --gres=gpu:1 --cpus-per-task=16 --mem=200G --time=08:00:00 \
#     --export=ALL,DELTA=$S/transfer_v1/dense_dpo_lightr1_llama8b/deltas/meta_llama_llama_3_1_8b_instruct_light_r1/deltas_step_500.pt,\
#OUT_DIR=$S/transfer_v1/oracle_masks_llama8b,DATASET=lightr1,OUT_TAG=deltafull,\
#REF_MASK=$S/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step500_sp97.5.pt \
#     scripts/mask_sweep_from_delta_lightr1.sbatch
#SBATCH --job-name=lr1_mask_sweep
#SBATCH --output=logs/lr1_mask_sweep_%j.out
#SBATCH --error=logs/lr1_mask_sweep_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --requeue

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

: "${DELTA:?Set DELTA=/scratch/.../deltas_step_500.pt}"
: "${OUT_DIR:?Set OUT_DIR=/scratch/.../oracle_masks_llama8b}"
: "${DATASET:?Set DATASET=lightr1}"
: "${OUT_TAG:?Set OUT_TAG=deltafull}"
REF_MASK="${REF_MASK:-}"
STEP="${STEP:-500}"

mkdir -p "$OUT_DIR"
if [ ! -f "$DELTA" ]; then echo "FATAL: DELTA not found: $DELTA"; exit 1; fi

# The work is on the CPU, so the GPU sits at 0% and would trip the RC 15-minute
# idle-GPU canceller. The sidecar keeps it visibly busy while real work proceeds --
# legitimate because the job IS working, never to hold an idle allocation.
python -c "import torch,time
gs=[torch.zeros(1024,1024,device=f'cuda:{i}') for i in range(torch.cuda.device_count())]
while True:
    for g in gs: (g@g).sum()
    torch.cuda.synchronize(); time.sleep(45)" &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

echo "=========================================="
echo "Light-R1 oracle mask sweep from delta"
echo "Job    : ${SLURM_JOB_ID:-local}  node ${SLURMD_NODENAME:-$(hostname)}"
echo "         cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "DELTA  = $DELTA"
echo "OUT_DIR= $OUT_DIR   tag=_src-${OUT_TAG}"
echo "REF    = ${REF_MASK:-<none>}"
echo "Started: $(date)"
echo "=========================================="

# Cheapest first: rho=99 keeps 80 M, rho=80 keeps 1.6 B, and the chunked top-k selector
# dominates runtime, so this order produces usable masks earliest.
#
# Override with e.g. SPARSITY_LIST="90 80" to split the sweep across jobs. Needed because
# the selector is memory-latency bound and runs ~6x slower on `zen` (first-gen EPYC)
# nodes than on `zen2`/`cascadelake`: measured rho=99 45:38 and rho=97.5 2:32:27 on
# c2205 against 10:04 and 24:28 for the Tulu3 half. At that rate rho=80 does not fit the
# cluster's 8 h hard cap at all, so it has to be moved, not just waited out.
# Splitting also avoids two jobs writing the same output path concurrently -- give each
# job a disjoint rho list.
read -r -a SPARSITIES <<< "${SPARSITY_LIST:-99 97.5 90 80}"
RC_ALL=0

for SP in "${SPARSITIES[@]}"; do
  OUT="$OUT_DIR/oracle_dpo_${DATASET}_step${STEP}_sp${SP}_src-${OUT_TAG}.pt"

  echo ""
  echo "########## rho=${SP}%  ->  $(basename "$OUT")  $(date) ##########"

  if [ -n "$REF_MASK" ] && [ "$REF_MASK" = "$OUT" ]; then
    echo "FATAL: REF_MASK equals OUT ($OUT); the gate would compare a file to itself."
    RC_ALL=1
    continue
  fi

  if [ -f "$OUT" ]; then
    echo "Mask already exists, skipping generation: $OUT"
  else
    python - "$DELTA" "$SP" "$OUT" <<'PY'
import sys, torch, gc, os
sys.path.insert(0, os.getcwd())
from src.utils.mask_utils import (
    create_mask_from_scores_gpu_efficient, save_masks, pooling_metadata,
)

delta_path, sp, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
MIN_LAYER_KEEP_RATIO = 0.0025   # held constant across the sweep; rho is the only knob

print(f"Loading deltas: {delta_path}", flush=True)
scores = torch.load(delta_path, map_location="cpu")
print(f"  {len(scores)} tensors", flush=True)

# In place: never hold two copies of the 32 GB dict on this side of the call.
with torch.no_grad():
    for name in scores:
        scores[name] = scores[name].abs_()
gc.collect()

masks = create_mask_from_scores_gpu_efficient(
    scores, sp, device="cpu",
    local_pool=False, min_layer_keep_ratio=MIN_LAYER_KEEP_RATIO,
)
del scores; gc.collect()

for n in masks:
    masks[n] = masks[n].to(torch.bool)

metadata = {
    "method": "checkpoint_difference_ground_truth",
    "score_source": "delta_file",
    "delta_file": delta_path,
    "sparsity_percent": sp,
    "initial_model": "meta-llama/Llama-3.1-8B-Instruct",
    "mlp_only": False,
    "device": "cpu",
    **pooling_metadata(local_pool=False, min_layer_keep_ratio=MIN_LAYER_KEEP_RATIO),
}
save_masks(masks, out, metadata)

kept = sum(int(t.sum()) for t in masks.values())
tot = sum(t.numel() for t in masks.values())
print(f"\nkept {kept:,} / {tot:,} = {100*kept/tot:.4f}%  (rho = {100-100*kept/tot:.4f}%)", flush=True)
PY
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "rho=${SP} FAILED with exit $RC"
      RC_ALL=$RC
      continue
    fi
  fi

  # Gate. Only rho=97.5 has a surviving paper mask to compare against; its source dense
  # run was deleted, so this Jaccard is the only number that says whether the Light-R1
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
print(f"Jaccard = {inter/max(union,1):.4f}   (chance at rho=97.5 ~1.27e-2; Tulu3 half scored 0.9763)")
print("high -> the sweep extends the published point")
print("low  -> report the Light-R1 sweep self-contained; do not splice the paper number")
PY
  fi

  ls -lh "$OUT"
  echo "########## rho=${SP}% done $(date) ##########"
done

echo ""
echo "Finished: $(date)"
ls -lh "$OUT_DIR"/oracle_dpo_${DATASET}_step${STEP}_sp*_src-${OUT_TAG}.pt 2>/dev/null

# sacct State=COMPLETED is not proof the work happened (DO_NOT_REPEAT): propagate failure.
exit $RC_ALL

#!/bin/bash
# Dedicated job: optimizer.step() microbench (no fwd/bwd, no checkpoints, no deltas).
#
# Example (element-only — reproduces scratch.md / job 6593156-style .out):
#   cd /scratch/$USER/rl_casino/<repo> && mkdir -p logs
#   export SCRATCH_USER_ROOT="/scratch/$USER"
#   export ELEM_MASK="/scratch/$USER/rl_casino_masks/orch_lr1_grasp6_6376972/random_elem_meta-llama_Llama-3.1-8B-Instruct_light_r1_sp97.5pct_seed42.pt"
#   export STEPS=50 TRIM_FRAC=0.10 LR=5e-7 BLOCK_SIZE=32
#   # Tensor subset selection (matches the historical 97.5% row at ~525M params / ~1050MB BF16):
#   export MAX_TOTAL_NUMEL=525000000 MAX_TENSORS=64 SELECTION_ORDER=model_order CAP_BEHAVIOR=break
#   unset BLOCK_MASK; export RUN_BLOCK=0
#   sbatch scripts/h200_sparse_adamw_optstep_microbench.slurm
#
# Example (element + block):
#   export ELEM_MASK="..."
#   export BLOCK_MASK="..."
#   export RUN_BLOCK=1
#   sbatch scripts/h200_sparse_adamw_optstep_microbench.slurm
#
# Artifacts land in:
#   /scratch/$USER/rl_casino_optstep_microbench/$JOBID/{elem,block}/optimizer_step_microbench.(csv|md)
#
# IMPORTANT: matched-policy reruns must export MAX_TOTAL_NUMEL/MAX_TENSORS/SELECTION_ORDER/CAP_BEHAVIOR
# explicitly. Older runs that did not set these used the prior microbench defaults and produced a
# different effective parameter subset. The .out logs each run's effective settings so future tables
# cannot quietly mix policies.

# Explorer: multigpu usually shorter wait than gpu for single H200 (same as BSR bench scripts).
# Override: sbatch --partition=gpu scripts/h200_sparse_adamw_optstep_microbench.slurm
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=h200_optstep_mb
#SBATCH --output=logs/h200_optstep_mb_%j.out
#SBATCH --error=logs/h200_optstep_mb_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
DEFAULT_TRAIN_ENV="${SCRATCH_USER_ROOT}/conda_envs/rl_casino"
if [ -n "${TRAIN_PY:-}" ] && [ -x "$TRAIN_PY" ]; then
  :
elif [ -n "${TRAIN_ENV:-}" ] && [ -x "${TRAIN_ENV}/bin/python" ]; then
  TRAIN_PY="${TRAIN_ENV}/bin/python"
elif [ -x "${DEFAULT_TRAIN_ENV}/bin/python" ]; then
  TRAIN_ENV="${DEFAULT_TRAIN_ENV}"
  TRAIN_PY="${TRAIN_ENV}/bin/python"
else
  TRAIN_PY="python3"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_BASE="${OUT_BASE:-/scratch/${USER}/rl_casino_optstep_microbench/${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_BASE"

STEPS="${STEPS:-50}"
TRIM_FRAC="${TRIM_FRAC:-0.10}"
LR="${LR:-5e-7}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

# Tensor subset selection (defaults match the historical 97.5% row policy: model_order + break,
# MAX_TOTAL_NUMEL=525M ~= one 4096x128256 lm_head/embedding tensor in BF16 ≈ 1050MB).
MAX_TOTAL_NUMEL="${MAX_TOTAL_NUMEL:-525000000}"
MAX_TENSORS="${MAX_TENSORS:-64}"
SELECTION_ORDER="${SELECTION_ORDER:-model_order}"
CAP_BEHAVIOR="${CAP_BEHAVIOR:-break}"

ELEM_MASK="${ELEM_MASK:?set ELEM_MASK to an element mask .pt}"
BLOCK_MASK="${BLOCK_MASK:-}"
RUN_BLOCK="${RUN_BLOCK:-0}"

echo "OUT_BASE=$OUT_BASE"
echo "STEPS=$STEPS TRIM_FRAC=$TRIM_FRAC LR=$LR BLOCK_SIZE=$BLOCK_SIZE"
echo "MAX_TOTAL_NUMEL=$MAX_TOTAL_NUMEL MAX_TENSORS=$MAX_TENSORS SELECTION_ORDER=$SELECTION_ORDER CAP_BEHAVIOR=$CAP_BEHAVIOR"
echo "ELEM_MASK=$ELEM_MASK"
echo "BLOCK_MASK=$BLOCK_MASK RUN_BLOCK=$RUN_BLOCK"

"$TRAIN_PY" scripts/microbench_optimizer_step.py \
  --out-dir "$OUT_BASE/elem" \
  --mask_path "$ELEM_MASK" \
  --mask_label elem \
  --steps "$STEPS" \
  --trim_frac "$TRIM_FRAC" \
  --lr "$LR" \
  --block_size "$BLOCK_SIZE" \
  --device cuda --dtype bf16 --sync_cuda 1 \
  --run_dense_torch 1 --run_dense_8bit 1 --run_sparse 1 \
  --max_total_numel "$MAX_TOTAL_NUMEL" \
  --max_tensors "$MAX_TENSORS" \
  --selection_order "$SELECTION_ORDER" \
  --cap_behavior "$CAP_BEHAVIOR"

if [ "$RUN_BLOCK" = "1" ] && [ -n "$BLOCK_MASK" ]; then
  "$TRAIN_PY" scripts/microbench_optimizer_step.py \
    --out-dir "$OUT_BASE/block" \
    --mask_path "$BLOCK_MASK" \
    --mask_label block \
    --steps "$STEPS" \
    --trim_frac "$TRIM_FRAC" \
    --lr "$LR" \
    --block_size "$BLOCK_SIZE" \
    --device cuda --dtype bf16 --sync_cuda 1 \
    --run_dense_torch 1 --run_dense_8bit 1 --run_sparse 1 \
    --max_total_numel "$MAX_TOTAL_NUMEL" \
    --max_tensors "$MAX_TENSORS" \
    --selection_order "$SELECTION_ORDER" \
    --cap_behavior "$CAP_BEHAVIOR"
fi

echo "DONE. Results in: $OUT_BASE"


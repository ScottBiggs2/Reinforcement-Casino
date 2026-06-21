#!/bin/bash
# End-to-end job: generate a random element mask then immediately run the
# optimizer-step microbench.  No pre-existing mask required.
#
# The mask is generated on CPU (model loaded with device_map=cpu) while the
# job runs on the H200 node.  The microbench then uses the GPU.
#
# ── Quickstart ────────────────────────────────────────────────────────────────
#
#   cd /scratch/$USER/rl_casino/<repo> && mkdir -p logs
#
#   # Single sparsity level:
#   export MODEL="meta-llama/Llama-3.1-8B-Instruct"
#   export SPARSITY_PERCENT=97.5
#   export SEED=42
#   sbatch scripts/h200_optstep_mb_from_random_mask.slurm
#
#   # Flight of three sparsity levels (97.5 / 95 / 90) — independent jobs:
#   export MODEL="meta-llama/Llama-3.1-8B-Instruct" SEED=42
#   for SP in 97.5 95.0 90.0; do
#     export SPARSITY_PERCENT=$SP
#     sbatch scripts/h200_optstep_mb_from_random_mask.slurm
#   done
#
# ── Output layout ─────────────────────────────────────────────────────────────
#
#   /scratch/$USER/rl_casino_optstep_microbench/$JOBID/
#     mask/   random_elem_<model>_sp<SP>pct_seed<SEED>.pt
#     elem/   optimizer_step_microbench.csv   (speed phase — backward compat)
#     elem/   optimizer_step_memory.csv       (memory phase)
#     elem/   optimizer_step_microbench.md    (combined human-readable report)
#
# ── Key tunables ──────────────────────────────────────────────────────────────
#   SPARSITY_PERCENT  float  target sparsity % (default: 97.5)
#   SEED              int    RNG seed for reproducible masks (default: 42)
#   MODEL             str    HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)
#   STEPS             int    optimizer steps in speed phase (default: 50)
#   TRIM_FRAC         float  trim fraction for speed stats (default: 0.10)
#   MAX_TOTAL_NUMEL   int    param cap for benchmark subset (default: 525000000)
#   MAX_TENSORS       int    tensor count cap (default: 64)
#
# ── IMPORTANT: matched-policy reruns ──────────────────────────────────────────
#   Always export SEED, MAX_TOTAL_NUMEL, MAX_TENSORS, SELECTION_ORDER,
#   and CAP_BEHAVIOR explicitly so future table rows can be compared safely.
#   The .out log echoes all effective settings.

#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --job-name=optstep_mb_randmask
#SBATCH --output=logs/optstep_mb_randmask_%j.out
#SBATCH --error=logs/optstep_mb_randmask_%j.err

set -euo pipefail

# ── Repo root ─────────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# ── Python interpreter ────────────────────────────────────────────────────────
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

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
SEED="${SEED:-42}"

STEPS="${STEPS:-50}"
TRIM_FRAC="${TRIM_FRAC:-0.10}"
LR="${LR:-5e-7}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

MAX_TOTAL_NUMEL="${MAX_TOTAL_NUMEL:-525000000}"
MAX_TENSORS="${MAX_TENSORS:-64}"
SELECTION_ORDER="${SELECTION_ORDER:-model_order}"
CAP_BEHAVIOR="${CAP_BEHAVIOR:-break}"

# Sanitise model name for use in filenames (replace / with _).
MODEL_SANITIZED="${MODEL//\//_}"

OUT_BASE="${OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_optstep_microbench/${SLURM_JOB_ID:-local}}"
MASK_DIR="${OUT_BASE}/mask"
mkdir -p "$MASK_DIR"

# Sparsity tag for filename: 97.5 → sp97.5pct
SP_TAG="${SPARSITY_PERCENT}"
MASK_PATH="${MASK_DIR}/random_elem_${MODEL_SANITIZED}_sp${SP_TAG}pct_seed${SEED}.pt"

# ── Echo settings ─────────────────────────────────────────────────────────────
echo "============================================================"
echo " h200_optstep_mb_from_random_mask"
echo "============================================================"
echo "OUT_BASE          = $OUT_BASE"
echo "MODEL             = $MODEL"
echo "SPARSITY_PERCENT  = $SPARSITY_PERCENT"
echo "SEED              = $SEED"
echo "MASK_PATH         = $MASK_PATH"
echo "STEPS             = $STEPS  TRIM_FRAC=$TRIM_FRAC  LR=$LR  BLOCK_SIZE=$BLOCK_SIZE"
echo "MAX_TOTAL_NUMEL   = $MAX_TOTAL_NUMEL  MAX_TENSORS=$MAX_TENSORS"
echo "SELECTION_ORDER   = $SELECTION_ORDER  CAP_BEHAVIOR=$CAP_BEHAVIOR"
echo "TRAIN_PY          = $TRAIN_PY"
echo "============================================================"

# ── Step 1: Generate random element mask (CPU) ────────────────────────────────
echo ""
echo "=== Step 1: Generating random element mask ==="
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "$MODEL" \
  --sparsity_percent "$SPARSITY_PERCENT" \
  --output_file "$MASK_PATH" \
  --mask-granularity element \
  --seed "$SEED"

echo "Mask written to: $MASK_PATH"

# ── Step 2: Run optimizer-step microbench (GPU) ───────────────────────────────
echo ""
echo "=== Step 2: Optimizer-step microbench (Phase 1: Speed, Phase 2: Memory) ==="
"$TRAIN_PY" scripts/microbench_optimizer_step.py \
  --out-dir "${OUT_BASE}/elem" \
  --mask_path "$MASK_PATH" \
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

echo ""
echo "============================================================"
echo "DONE. Results in: ${OUT_BASE}/elem/"
echo "  Speed CSV : ${OUT_BASE}/elem/optimizer_step_microbench.csv"
echo "  Memory CSV: ${OUT_BASE}/elem/optimizer_step_memory.csv"
echo "  Report    : ${OUT_BASE}/elem/optimizer_step_microbench.md"
echo "============================================================"

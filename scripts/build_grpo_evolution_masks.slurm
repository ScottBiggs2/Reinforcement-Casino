#!/usr/bin/env bash
# Build the 5 masks for the GRPO "subnetwork evolution" study, at 97.5% sparsity:
#   - magnitude masks (accumulated Σ|Δθ| up to step T) for T in {50,100,250}
#   - oracle mask (checkpoint-diff |θ_500 − θ_0|) from dense checkpoint-500
#   - random elementwise baseline (seed 42)
# from a dense GRPO run produced by scripts/grpo_openr1_llama31_slurm.sh with delta logging
# (GRPO_DELTA_LOG_INTERVAL=50, GRPO_DELTA_LOG_END_STEP>=250).
#
# GRPO run layout (flat; see src/full_training/GRPO_train.py:224-228):
#   <GRPO_DENSE_OUTPUT_BASE>/<slug>/deltas/deltas_step_<N>.pt , base_state.pt
#   <GRPO_DENSE_OUTPUT_BASE>/<slug>/checkpoints/checkpoint-<STEP>/
#
# Drop-in (repo root on Explorer), after the dense run has finished:
#   export HF_TOKEN="$(cat /scratch/$USER/hf_home/token)"   # gated Llama initial_model (cached, but be safe)
#   sbatch scripts/build_grpo_evolution_masks.slurm
#
# Overrides:
#   GRPO_DENSE_RUN_SLUG (default grpo_dense_openr1_steps500_evolstudy)
#   MAG_STEPS="50 100 250"   CHECKPOINT_STEP=500   SPARSITY_PERCENT=97.5
#   MIN_LAYER_KEEP_RATIO=0.0025   RANDOM_SEED=42   MODEL=meta-llama/Llama-3.1-8B-Instruct
#   MASK_OUT_DIR=<dir>       (default <scratch>/rl_casino_grpo/masks/evolstudy_<jobid>)
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=grpo_evol_masks
#SBATCH --output=logs/grpo_evol_masks_%j.out
#SBATCH --error=logs/grpo_evol_masks_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# --- Environment (mirror the GRPO orchestrator's setup; no DPO pipeline_common needed) ---
SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-}"
# Oracle (checkpoint_diff, loads base model) and random (reads model param shapes) need the
# HF cache that holds the gated Llama snapshot. Default to the scratch cache used elsewhere.
export HF_HOME="${HF_HOME:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/hub}"

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
  exit 1
fi

# --- Parameters ---
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MAG_STEPS="${MAG_STEPS:-50 100 250}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-500}"

GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
GRPO_DENSE_RUN_SLUG="${GRPO_DENSE_RUN_SLUG:-grpo_dense_openr1_steps500_evolstudy}"
RUN_DIR="${GRPO_DENSE_OUTPUT_BASE}/${GRPO_DENSE_RUN_SLUG}"
DELTA_DIR="${RUN_DIR}/deltas"
CKPT_DIR="${RUN_DIR}/checkpoints/checkpoint-${CHECKPOINT_STEP}"

MASK_OUT_DIR="${MASK_OUT_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks/evolstudy_${SLURM_JOB_ID:-local}}"
mkdir -p "$MASK_OUT_DIR"

echo "================================================================="
echo " GRPO evolution masks — build"
echo "================================================================="
echo "RUN_DIR       = ${RUN_DIR}"
echo "DELTA_DIR     = ${DELTA_DIR}"
echo "CKPT_DIR      = ${CKPT_DIR}"
echo "MASK_OUT_DIR  = ${MASK_OUT_DIR}"
echo "MAG_STEPS     = ${MAG_STEPS}"
echo "SPARSITY      = ${SPARSITY_PERCENT}%  min_layer_keep_ratio=${MIN_LAYER_KEEP_RATIO}"
echo "================================================================="

# --- Preflight: required delta snapshots + checkpoint present ---
if [ ! -d "$DELTA_DIR" ]; then
  echo "ERROR: delta dir not found: $DELTA_DIR (did the dense run log deltas?)" >&2
  exit 2
fi
for T in $MAG_STEPS; do
  if [ ! -f "${DELTA_DIR}/deltas_step_${T}.pt" ]; then
    echo "ERROR: missing ${DELTA_DIR}/deltas_step_${T}.pt (need every 50-step snapshot up to each target)." >&2
    ls -la "$DELTA_DIR" >&2 || true
    exit 2
  fi
done
if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: checkpoint dir not found: $CKPT_DIR" >&2
  exit 2
fi

SUITE_LIST="${MASK_OUT_DIR}/suite_list.txt"
: > "$SUITE_LIST"
LABELS=()

# --- (1) Magnitude masks at each target step (accumulated |Δθ|; matches paper's mag_step200) ---
for T in $MAG_STEPS; do
  OUT="${MASK_OUT_DIR}/mag_step${T}.pt"
  echo ""; echo "--- magnitude mask @ step ${T} -> ${OUT} ---"
  "$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "$DELTA_DIR" \
    --method magnitude \
    --sparsity_percent "$SPARSITY_PERCENT" \
    --target_step "$T" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT"
  [ -f "$OUT" ] || { echo "ERROR: magnitude mask not written: $OUT" >&2; exit 1; }
  echo "$OUT" >> "$SUITE_LIST"
  LABELS+=( "mag_step${T}" )
done

# --- (2) Oracle mask (checkpoint-diff |θ_500 − θ_0|) ---
ORACLE_OUT="${MASK_OUT_DIR}/oracle_gt.pt"
echo ""; echo "--- oracle checkpoint-diff mask (ckpt-${CHECKPOINT_STEP}) -> ${ORACLE_OUT} ---"
"$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
  --initial_model "$MODEL" \
  --final_model "$CKPT_DIR" \
  --sparsity_percent "$SPARSITY_PERCENT" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "$ORACLE_OUT"
[ -f "$ORACLE_OUT" ] || { echo "ERROR: oracle mask not written: $ORACLE_OUT" >&2; exit 1; }
echo "$ORACLE_OUT" >> "$SUITE_LIST"
LABELS+=( "oracle_gt" )

# --- (3) Random elementwise baseline (seed 42) ---
RAND_OUT="${MASK_OUT_DIR}/random.pt"
echo ""; echo "--- random elementwise baseline -> ${RAND_OUT} ---"
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "$MODEL" \
  --sparsity_percent "$SPARSITY_PERCENT" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --seed "$RANDOM_SEED" \
  --mask-granularity element \
  --output_file "$RAND_OUT"
[ -f "$RAND_OUT" ] || { echo "ERROR: random mask not written: $RAND_OUT" >&2; exit 1; }
echo "$RAND_OUT" >> "$SUITE_LIST"
LABELS+=( "random" )

echo ""
echo "================================================================="
echo "✓ Masks built. suite_list.txt:"
cat "$SUITE_LIST"
echo ""
echo "Suggested labels (order matches suite_list):"
echo "  MASK_SUITE_LABELS=\"${LABELS[*]}\""
echo ""
ls -lh "$MASK_OUT_DIR"
echo "================================================================="

#!/bin/bash
# Build the 97.5% masks for Olmo-3-7B / light-r1 from a completed DENSE run.
#
# Produces three element-wise masks (same granularity/sparsity as the random mask,
# so all feed sparse_dpo_efficiency.py --block_size 32 identically):
#   1. magnitude@250   — warm "rewinding" mask (|Δθ| accumulated by step 250)
#   2. oracle_deltalog — end-to-end oracle via delta logs (|Δθ| at final step 500)
#   3. oracle_ckptdiff — end-to-end oracle via checkpoint diff (base HF vs step-500 ckpt)
#
# (2) and (3) are two independent ways to build the "oracle"; we keep whichever
# yields a well-conditioned mask. Both difference bf16 weights, so for a low-LR /
# short (500-step) run either can come out coarse — the magnitude@250 mask is the
# fallback (that is what the Qwen3 "oracle" run ultimately shipped).
#
# REQUIRED: DENSE_RUN_ID must point at the dense run's RUN_ID (the dir under
# ${TRAIN_OUT_BASE}). Submit from repo root:
#   sbatch --export=ALL,DENSE_RUN_ID=<run_id> scripts/gen_masks_olmo3_7b_lr1.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --job-name=mk_masks_olmo3_7b
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mk_masks_olmo3_7b_%j.out
#SBATCH --error=logs/mk_masks_olmo3_7b_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
export MODEL="${MODEL:-allenai/Olmo-3-7B-Instruct}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
SPARSITY_PCT="${SPARSITY_PCT:-97.5}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
MAG_TARGET_STEP="${MAG_TARGET_STEP:-250}"     # warm "rewinding" mask step
ORACLE_TARGET_STEP="${ORACLE_TARGET_STEP:-500}"  # end-to-end delta-log oracle step

if [ -z "${DENSE_RUN_ID:-}" ]; then
  echo "ERROR: DENSE_RUN_ID is required." >&2
  echo "       sbatch --export=ALL,DENSE_RUN_ID=<run_id> scripts/gen_masks_olmo3_7b_lr1.slurm" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Pipeline boilerplate (paths, env, score device)
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

# sanitize_model_name(MODEL)_<ds sanitized> — must match DPO_train.py output tree.
SUBDIR="allenai_olmo_3_7b_instruct_light_r1"
DELTA_DIR="${TRAIN_OUT_BASE}/${DENSE_RUN_ID}/deltas/${SUBDIR}"
CKPT_PARENT="${TRAIN_OUT_BASE}/${DENSE_RUN_ID}/checkpoints/${SUBDIR}"

# Resolve the final HF checkpoint dir (prefer checkpoint-500, else newest checkpoint-*).
FINAL_CKPT=""
if [ -d "${CKPT_PARENT}/checkpoint-500" ]; then
  FINAL_CKPT="${CKPT_PARENT}/checkpoint-500"
elif [ -d "${CKPT_PARENT}" ]; then
  FINAL_CKPT="$(ls -d "${CKPT_PARENT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
fi

MASK_DIR="${MASK_OUT_BASE}/olmo3_7b_light_r1"
mkdir -p "$MASK_DIR" logs

MAG_OUT="${MASK_DIR}/warm_magnitude_${SUBDIR}_sp${SPARSITY_PCT}pct_step${MAG_TARGET_STEP}.pt"
ORACLE_DL_OUT="${MASK_DIR}/oracle_deltalog_${SUBDIR}_sp${SPARSITY_PCT}pct_step${ORACLE_TARGET_STEP}.pt"
ORACLE_CD_OUT="${MASK_DIR}/oracle_ckptdiff_${SUBDIR}_sp${SPARSITY_PCT}pct.pt"

echo "===== Mask generation: Olmo-3-7B / light-r1 (DENSE_RUN_ID=${DENSE_RUN_ID}) ====="
echo "  delta dir  : ${DELTA_DIR}"
echo "  final ckpt : ${FINAL_CKPT:-<none found>}"
echo "  mask dir   : ${MASK_DIR}"
if [ ! -d "${DELTA_DIR}" ]; then
  echo "ERROR: delta dir not found: ${DELTA_DIR}" >&2
  exit 1
fi
ls "${DELTA_DIR}/" || true

# Each step is non-fatal: one failing method must not block the others.
fail=0
run_step() {
  local desc="$1"; shift
  echo ""
  echo "---- ${desc} ----"
  set +e
  "$@"
  local ec=$?
  set -e
  if [ "$ec" -ne 0 ]; then
    echo "WARNING: ${desc} failed (exit ${ec}). Continuing." >&2
    fail=$((fail + 1))
  else
    echo "OK: ${desc}"
  fi
}

# ---------------------------------------------------------------------------
# 1. Warm magnitude mask (step 250) — "rewinding" mask
# ---------------------------------------------------------------------------
run_step "magnitude @ step ${MAG_TARGET_STEP}" \
  "$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
    --delta_log_dir        "$DELTA_DIR" \
    --method               magnitude \
    --sparsity_percent     "$SPARSITY_PCT" \
    --target_step          "$MAG_TARGET_STEP" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file          "$MAG_OUT"

# ---------------------------------------------------------------------------
# 2. Oracle A — delta-log ground_truth (|Δθ| at final step)
# ---------------------------------------------------------------------------
run_step "oracle (delta-log ground_truth) @ step ${ORACLE_TARGET_STEP}" \
  "$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
    --delta_log_dir        "$DELTA_DIR" \
    --method               ground_truth \
    --sparsity_percent     "$SPARSITY_PCT" \
    --target_step          "$ORACLE_TARGET_STEP" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file          "$ORACLE_DL_OUT"

# ---------------------------------------------------------------------------
# 3. Oracle B — checkpoint diff (base HF weights vs step-500 checkpoint)
# ---------------------------------------------------------------------------
if [ -n "${FINAL_CKPT}" ] && [ -d "${FINAL_CKPT}" ]; then
  run_step "oracle (checkpoint diff) base vs ${FINAL_CKPT##*/}" \
    "$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
      --initial_model        "$MODEL" \
      --final_model          "$FINAL_CKPT" \
      --sparsity_percent     "$SPARSITY_PCT" \
      --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
      --output_file          "$ORACLE_CD_OUT"
else
  echo "WARNING: no final checkpoint found under ${CKPT_PARENT}; skipping checkpoint-diff oracle." >&2
  fail=$((fail + 1))
fi

echo ""
echo "===== Mask generation done (${fail} step(s) failed) ====="
echo "  magnitude@${MAG_TARGET_STEP}     : ${MAG_OUT}"
echo "  oracle (delta-log)  : ${ORACLE_DL_OUT}"
echo "  oracle (ckpt-diff)  : ${ORACLE_CD_OUT}"
echo ""
echo "  Then sparse-train on each mask, e.g.:"
echo "    sbatch --export=ALL,MASK_FILE=${ORACLE_DL_OUT},MASK_LABEL=oracle_deltalog scripts/sparse_dpo_olmo3_7b_masked_lr1_500.slurm"

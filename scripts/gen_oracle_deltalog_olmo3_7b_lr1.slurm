#!/bin/bash
# Build delta-log "ground_truth" ORACLE masks for Olmo-3-7B / light-r1 at one or
# more sparsities, from a completed DENSE run's final-step (500) deltas.
#
# Lean companion to gen_masks_olmo3_7b_lr1.slurm: this builds ONLY the delta-log
# oracle (|Δθ| at the final step), at each sparsity in SPARSITY_LIST. Default is
# 90 and 99 (the 97.5 oracle is produced by gen_masks_olmo3_7b_lr1.slurm).
#
# REQUIRED: DENSE_RUN_ID (full RUN_ID like 20260728_184922_8811836, or bare jobid).
# Submit from repo root:
#   sbatch --export=ALL,DENSE_RUN_ID=<run_id> scripts/gen_oracle_deltalog_olmo3_7b_lr1.slurm
#   # override sparsities:
#   sbatch --export=ALL,DENSE_RUN_ID=<run_id>,SPARSITY_LIST="90 99" scripts/gen_oracle_deltalog_olmo3_7b_lr1.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --job-name=mk_oracle_dl_olmo3
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mk_oracle_dl_olmo3_%j.out
#SBATCH --error=logs/mk_oracle_dl_olmo3_%j.err

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
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
ORACLE_TARGET_STEP="${ORACLE_TARGET_STEP:-500}"
# Space-separated list of sparsity percentages to build.
# NOTE: captured into a DIFFERENTLY-named var *before* sourcing pipeline_common.sh,
# which hard-sets SPARSITY_LIST=("97.5") and would otherwise clobber this value.
ORACLE_SPARSITY_LIST="${SPARSITY_LIST:-90 99}"

if [ -z "${DENSE_RUN_ID:-}" ]; then
  echo "ERROR: DENSE_RUN_ID is required." >&2
  echo "       sbatch --export=ALL,DENSE_RUN_ID=<run_id> scripts/gen_oracle_deltalog_olmo3_7b_lr1.slurm" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Pipeline boilerplate (paths, env, score device)
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

# Resolve DENSE_RUN_ID: accept the full RUN_ID (e.g. 20260728_184922_8811836) OR a
# bare Slurm jobid (e.g. 8811836) — the training RUN_ID is <timestamp>_<jobid>.
if [ ! -d "${TRAIN_OUT_BASE}/${DENSE_RUN_ID}" ]; then
  shopt -s nullglob
  _matches=( "${TRAIN_OUT_BASE}"/*"${DENSE_RUN_ID}" )
  shopt -u nullglob
  if [ "${#_matches[@]}" -eq 1 ]; then
    DENSE_RUN_ID="$(basename "${_matches[0]}")"
    echo "NOTE: resolved DENSE_RUN_ID → ${DENSE_RUN_ID}"
  elif [ "${#_matches[@]}" -eq 0 ]; then
    echo "ERROR: no train dir under ${TRAIN_OUT_BASE} matching '*${DENSE_RUN_ID}'" >&2
    exit 1
  else
    echo "ERROR: DENSE_RUN_ID='${DENSE_RUN_ID}' is ambiguous; matches:" >&2
    printf '       %s\n' "${_matches[@]}" >&2
    exit 1
  fi
fi

SUBDIR="allenai_olmo_3_7b_instruct_light_r1"
DELTA_DIR="${TRAIN_OUT_BASE}/${DENSE_RUN_ID}/deltas/${SUBDIR}"
MASK_DIR="${MASK_OUT_BASE}/olmo3_7b_light_r1"
mkdir -p "$MASK_DIR" logs

echo "===== Delta-log oracle masks: Olmo-3-7B / light-r1 (DENSE_RUN_ID=${DENSE_RUN_ID}) ====="
echo "  delta dir     : ${DELTA_DIR}"
echo "  target step   : ${ORACLE_TARGET_STEP}"
echo "  sparsities    : ${ORACLE_SPARSITY_LIST}"
echo "  mask dir      : ${MASK_DIR}"
if [ ! -f "${DELTA_DIR}/deltas_step_${ORACLE_TARGET_STEP}.pt" ]; then
  echo "ERROR: missing ${DELTA_DIR}/deltas_step_${ORACLE_TARGET_STEP}.pt" >&2
  exit 1
fi

fail=0
for SP in ${ORACLE_SPARSITY_LIST}; do
  OUT="${MASK_DIR}/oracle_deltalog_${SUBDIR}_sp${SP}pct_step${ORACLE_TARGET_STEP}.pt"
  echo ""
  echo "---- oracle (delta-log ground_truth) sparsity=${SP}% ----"
  set +e
  "$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
    --delta_log_dir        "$DELTA_DIR" \
    --method               ground_truth \
    --sparsity_percent     "$SP" \
    --target_step          "$ORACLE_TARGET_STEP" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file          "$OUT"
  ec=$?
  set -e
  if [ "$ec" -ne 0 ]; then
    echo "WARNING: sparsity=${SP}% failed (exit ${ec})." >&2
    fail=$((fail + 1))
  else
    echo "OK: ${OUT}"
  fi
done

echo ""
echo "===== Done (${fail} sparsity step(s) failed) ====="
for SP in ${ORACLE_SPARSITY_LIST}; do
  echo "  sp${SP}: ${MASK_DIR}/oracle_deltalog_${SUBDIR}_sp${SP}pct_step${ORACLE_TARGET_STEP}.pt"
done
echo ""
echo "  Then sparse-train each, e.g.:"
echo "    sbatch --export=ALL,MASK_FILE=${MASK_DIR}/oracle_deltalog_${SUBDIR}_sp90pct_step${ORACLE_TARGET_STEP}.pt,MASK_LABEL=oracle_deltalog_sp90 scripts/sparse_dpo_olmo3_7b_masked_lr1_500.slurm"

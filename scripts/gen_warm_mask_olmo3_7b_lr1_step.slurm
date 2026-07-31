#!/bin/bash
# Build ONE warm ("magnitude rewinding") mask for Olmo-3-7B / light-r1 at a given step k.
#
# This is the single-method sibling of scripts/gen_masks_olmo3_7b_lr1.slurm, which builds three
# masks (magnitude + delta-log oracle + checkpoint-diff oracle) in one job. For the k-sweep we
# only want the magnitude arm, one k per job, so the two builds can run in parallel and neither
# pays for the ~45 min oracle rebuild.
#
# Two deliberate differences from gen_masks_olmo3_7b_lr1.slurm:
#   1. only --method magnitude runs;
#   2. failures are FATAL. That script's run_step() swallows a non-zero exit and still ends the
#      job successfully, which is the right call when three independent methods share a job and
#      wrong here, where the one step IS the job.
#
# Output naming is byte-identical to gen_masks_olmo3_7b_lr1.slurm's MAG_OUT, so masks produced
# here are indistinguishable from the k=250 mask already in the mask dir.
#
# REQUIRED: DENSE_RUN_ID (dense run's RUN_ID under ${TRAIN_OUT_BASE}) and MAG_TARGET_STEP.
# Submit from repo root:
#   sbatch --export=ALL,DENSE_RUN_ID=20260728_184922_8811836,MAG_TARGET_STEP=50 \
#          scripts/gen_warm_mask_olmo3_7b_lr1_step.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --job-name=mk_warm_olmo3_7b
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mk_warm_olmo3_7b_%j.out
#SBATCH --error=logs/mk_warm_olmo3_7b_%j.err

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

if [ -z "${DENSE_RUN_ID:-}" ]; then
  echo "ERROR: DENSE_RUN_ID is required." >&2
  exit 1
fi
# No default for k on purpose: gen_masks_olmo3_7b_lr1.slurm defaults MAG_TARGET_STEP=250, and
# silently rebuilding the k=250 mask is exactly the mistake this script exists to avoid.
if [ -z "${MAG_TARGET_STEP:-}" ]; then
  echo "ERROR: MAG_TARGET_STEP is required (no default; set it explicitly)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Pipeline boilerplate (paths, env, score device)
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

# Resolve DENSE_RUN_ID: accept the full RUN_ID (e.g. 20260728_184922_8811836) OR a bare
# Slurm jobid (e.g. 8811836) — the training RUN_ID is <timestamp>_<jobid>.
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

# sanitize_model_name(MODEL)_<ds sanitized> — must match DPO_train.py output tree.
SUBDIR="allenai_olmo_3_7b_instruct_light_r1"
DELTA_DIR="${TRAIN_OUT_BASE}/${DENSE_RUN_ID}/deltas/${SUBDIR}"

MASK_DIR="${MASK_OUT_BASE}/olmo3_7b_light_r1"
mkdir -p "$MASK_DIR" logs

MAG_OUT="${MASK_DIR}/warm_magnitude_${SUBDIR}_sp${SPARSITY_PCT}pct_step${MAG_TARGET_STEP}.pt"

echo "===== Warm magnitude mask: Olmo-3-7B / light-r1 @ step ${MAG_TARGET_STEP} ====="
echo "  dense run  : ${DENSE_RUN_ID}"
echo "  delta dir  : ${DELTA_DIR}"
echo "  sparsity   : ${SPARSITY_PCT}%  min_layer_keep_ratio=${MIN_LAYER_KEEP_RATIO}"
echo "  output     : ${MAG_OUT}"

if [ ! -d "${DELTA_DIR}" ]; then
  echo "ERROR: delta dir not found: ${DELTA_DIR}" >&2
  exit 1
fi
if [ ! -f "${DELTA_DIR}/deltas_step_${MAG_TARGET_STEP}.pt" ]; then
  echo "ERROR: no snapshot for step ${MAG_TARGET_STEP} in ${DELTA_DIR}" >&2
  ls "${DELTA_DIR}/" >&2 || true
  exit 1
fi
if [ -f "${MAG_OUT}" ]; then
  echo "ERROR: ${MAG_OUT} already exists; refusing to overwrite." >&2
  echo "       Remove it or change SPARSITY_PCT/MAG_TARGET_STEP." >&2
  exit 1
fi

"$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir        "$DELTA_DIR" \
  --method               magnitude \
  --sparsity_percent     "$SPARSITY_PCT" \
  --target_step          "$MAG_TARGET_STEP" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file          "$MAG_OUT"

echo ""
echo "===== Done. Mask: ${MAG_OUT} ====="
echo "  Verify before training:"
echo "    \$TRAIN_PY src/utils/verify_mask_coverage.py --model_name ${MODEL} --mask_file ${MAG_OUT}"
echo "  Then:"
echo "    sbatch --export=ALL,MASK_FILE=${MAG_OUT},MASK_LABEL=magnitude_k${MAG_TARGET_STEP} \\"
echo "           scripts/sparse_dpo_olmo3_7b_masked_lr1_500.slurm"

#!/usr/bin/env bash
# One stage of the certifiability-margin pipeline (src/analysis/certifiability_margins.py).
#
# Stages, set via CERT_STAGE:
#   cache_only  build magnitude_caches/mag_aggregate_step_<k>.pt        (streams every delta once)
#   tau         tau_hat for every arm x rho -> certifiability_tau.json  (needs the caches)
#   arm         one arm's margins/gaps/cert -> arm_shards/arm_<name>.pt (Slurm array, see CERT_ARMS)
#   merge       combine shards -> CSV/NPZ/JSON, then figures            (no checkpoint load, cheap)
#
# Driver: scripts/submit_certifiability_margins.sh (submits the whole DAG for a cell).
#
# HARD RULE: OUT_DIR must not be under $HOME. The magnitude caches are ~30 GB each and
# Home quotas are typically small; filling them wedges the account. Checked here and again in
# Python (certifiability_margins.guard_not_under_home).
#
#SBATCH --job-name=certmargin
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/certifiability_%x_%j.out
#SBATCH --error=logs/certifiability_%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: python not found at ${PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-${SCRATCH_USER_ROOT}/hf_cache/hub}"
# Threads: the histogram/bucketize passes are the hot loop and scale with cpus-per-task.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

: "${OUT_DIR:?set OUT_DIR (must be under /scratch, not \$HOME)}"
: "${INITIAL_MODEL:?set INITIAL_MODEL (prefer base_state.pt from the run deltas dir)}"
: "${FINAL_MODEL:?set FINAL_MODEL (final HF checkpoint dir)}"
: "${DELTA_LOG_DIR:?set DELTA_LOG_DIR}"

case "$(cd "$(dirname "$OUT_DIR")" 2>/dev/null && pwd)/$(basename "$OUT_DIR")" in
  "${HOME}"|"${HOME}"/*)
    echo "REFUSING: OUT_DIR=${OUT_DIR} is under \$HOME (${HOME})." >&2
    echo "  Magnitude caches are ~30 GB each. Use ${SCRATCH_USER_ROOT}/rl_casino_analysis/... instead." >&2
    exit 1
    ;;
esac

STAGE="${CERT_STAGE:-tau}"
MILESTONES="${MILESTONES:-50,100,150,200}"
SPARSITIES="${SPARSITIES:-97.5}"
RANDOM_SEED="${RANDOM_SEED:-42}"
TAU_RULE="${TAU_RULE:-hybrid_global_phase}"
HYBRID_RATIO="${HYBRID_RATIO:-0.0025}"
TIE_BREAK_SCALE="${TIE_BREAK_SCALE:-1e-12}"
SCORE_SCOPE="${SCORE_SCOPE:-builder_2d}"
SCOPE_SOURCE="${SCOPE_SOURCE:-initial}"
CHECKPOINT_DTYPE="${CHECKPOINT_DTYPE:-bfloat16}"
WORK_DTYPE="${WORK_DTYPE:-float64}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-2000}"
KTH_NUM_BINS="${KTH_NUM_BINS:-4096}"
MODEL_LABEL="${MODEL_LABEL:-}"
DATASET_LABEL="${DATASET_LABEL:-}"
OBJECTIVE_LABEL="${OBJECTIVE_LABEL:-}"

mkdir -p "$OUT_DIR"

ARGS=(
  src/analysis/certifiability_margins.py
  --initial_model "$INITIAL_MODEL"
  --final_model "$FINAL_MODEL"
  --delta_log_dir "$DELTA_LOG_DIR"
  --out_dir "$OUT_DIR"
  --magnitude_milestones "$MILESTONES"
  --sparsity_percents "$SPARSITIES"
  --random_seed "$RANDOM_SEED"
  --tau_rule "$TAU_RULE"
  --hybrid_min_layer_keep_ratio "$HYBRID_RATIO"
  --tie_break_relative_scale "$TIE_BREAK_SCALE"
  --score_scope "$SCORE_SCOPE"
  --scope_source "$SCOPE_SOURCE"
  --checkpoint_dtype "$CHECKPOINT_DTYPE"
  --work_dtype "$WORK_DTYPE"
  --histogram_bins "$HISTOGRAM_BINS"
  --kth_num_bins "$KTH_NUM_BINS"
  --model_label "$MODEL_LABEL"
  --dataset_label "$DATASET_LABEL"
  --objective_label "$OBJECTIVE_LABEL"
)
if [ "${FORCE_MAG_CACHE_REBUILD:-0}" = "1" ]; then
  ARGS+=( --force_magnitude_cache_rebuild )
fi

echo "stage=${STAGE} out_dir=${OUT_DIR}"
echo "milestones=${MILESTONES} sparsities=${SPARSITIES} tau_rule=${TAU_RULE} tie_break=${TIE_BREAK_SCALE}"

case "$STAGE" in
  cache_only|tau|merge)
    "$PY" "${ARGS[@]}" --execution_mode "$STAGE"
    ;;
  arm)
    : "${CERT_ARMS:?set CERT_ARMS as a comma-separated arm list for the array}"
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
      echo "ERROR: stage=arm requires a Slurm array job" >&2
      exit 1
    fi
    IFS=',' read -r -a ARMS <<< "$CERT_ARMS"
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ARMS[@]}" ]; then
      echo "ERROR: array id ${SLURM_ARRAY_TASK_ID} out of range for ${#ARMS[@]} arms" >&2
      exit 1
    fi
    ARM="${ARMS[$SLURM_ARRAY_TASK_ID]// /}"
    echo "arm=${ARM} (array ${SLURM_ARRAY_TASK_ID})"
    "$PY" "${ARGS[@]}" --execution_mode arm --arm "$ARM"
    ;;
  *)
    echo "ERROR: unknown CERT_STAGE=${STAGE}" >&2
    exit 1
    ;;
esac

if [ "$STAGE" = "merge" ]; then
  "$PY" scripts/plot_certifiability_margins.py --analysis-dir "$OUT_DIR"
fi

echo "Done stage ${STAGE}."

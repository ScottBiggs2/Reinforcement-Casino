#!/usr/bin/env bash
# Parallel mask score-gap pipeline (milestone shards). Does NOT modify slurm_mask_score_gap_light_r1.slurm.
#
# Submit stages from the login node with scripts/submit_mask_score_gap_parallel_light_r1.sh (one-shot DAG).
# That script skips the cache_only Slurm job when OUT_DIR/magnitude_caches/mag_aggregate_step_*.pt are all present
# (e.g. built by slurm_mask_score_gap_light_r1.sh); export OUT_DIR before invoking it on the login node.
# Progress: OUT_DIR=... JOBID=<id> bash scripts/check_mask_score_gap_status.sh
#
# Or run a single stage manually (examples):
#   export MASK_GAP_STAGE=cache_only OUT_DIR=/scratch/$USER/mgap_par
#   sbatch scripts/slurm_mask_score_gap_parallel_light_r1.slurm
#
#   export MASK_GAP_STAGE=baseline_shard
#   sbatch --dependency=afterok:<cache_jobid> scripts/slurm_mask_score_gap_parallel_light_r1.slurm
#
#   export MASK_GAP_STAGE=milestone_shard MAGNITUDE_MILESTONES=50,100,150,200
#   sbatch --array=0-3 scripts/slurm_mask_score_gap_parallel_light_r1.slurm
#
#   export MASK_GAP_STAGE=merge_shards
#   sbatch --dependency=afterok:<array_jobid> scripts/slurm_mask_score_gap_parallel_light_r1.slurm
#
# Do not rsync magnitude_caches/ into $HOME/repo — copy only CSV/NPZ/JSON/figures after merge.
#
# Baseline #SBATCH (manual sbatch only). scripts/submit_mask_score_gap_parallel_light_r1.sh passes per-stage
# --mem/--time/--cpus-per-task so merge_shards does not request heavy-node RAM; see MGAP_SBATCH_* there.
# Raise limits if sacct shows OOM or TIMEOUT (checkpoint-heavy stages may need 256G like the monolithic script).
# Default HISTOGRAM_BINS=1024 saves CPU vs 2048; export HISTOGRAM_BINS=512|2048 to tune.
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --job-name=mgap_par
#SBATCH --output=logs/mask_score_gap_parallel_%j.out
#SBATCH --error=logs/mask_score_gap_parallel_%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
_DEFAULT_TRAIN_ENV="${SCRATCH_USER_ROOT}/conda_envs/rl_casino"
TRAIN_ENV="${TRAIN_ENV:-${_DEFAULT_TRAIN_ENV}}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  TRAIN_ENV="${_DEFAULT_TRAIN_ENV}"
  TRAIN_PY="${TRAIN_ENV}/bin/python"
fi
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Headless nodes: default Qt/GTK backends often yield no PNGs; Agg writes figures reliably.
export MPLBACKEND="${MPLBACKEND:-Agg}"

export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-${SCRATCH_USER_ROOT}/hf_cache/hub}"
_HF_DEFAULT_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  if [ -n "${HF_TOKEN_FILE:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
    export HF_TOKEN="$(tr -d '\n\r' < "${HF_TOKEN_FILE}")"
  elif [ -f "${_HF_DEFAULT_TOKEN_FILE}" ]; then
    export HF_TOKEN="$(tr -d '\n\r' < "${_HF_DEFAULT_TOKEN_FILE}")"
  fi
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CKPT500_DIR="${CKPT500_DIR:-${SCRATCH_USER_ROOT}/rl_casino_train/dpo5k_dense_light-r1/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500}"
DELTA_LOG_DIR="${DELTA_LOG_DIR:-${SCRATCH_USER_ROOT}/rl_casino_train/dpo5k_dense_light-r1/deltas/meta_llama_llama_3_1_8b_instruct_light_r1}"
OUT_DIR="${OUT_DIR:-${SCRATCH_USER_ROOT}/rl_casino_analysis/mask_score_gap_parallel/${SLURM_JOB_ID:-local}}"
MAGNITUDE_MILESTONES="${MAGNITUDE_MILESTONES:-50,100,150,200}"
FORCE_MAG_CACHE_REBUILD="${FORCE_MAG_CACHE_REBUILD:-0}"
RANDOM_SEEDS="${RANDOM_SEEDS:-}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-1024}"
MAX_KEYS="${MAX_KEYS:-}"
VERIFY_DELTAS_DIR="${VERIFY_DELTAS_DIR:-}"
SPARSITY_PERCENT="${CERT_SPARSITY_PERCENT:-${SPARSITY_PERCENT:-97.5}}"
CERT_MIN_LAYER_KEEP_RATIO="${CERT_MIN_LAYER_KEEP_RATIO:-0}"
CERT_MATCH_TIE_BREAK="${CERT_MATCH_TIE_BREAK:-1}"
CHECKPOINT_DTYPE="${MASK_GAP_CHECKPOINT_DTYPE:-${CHECKPOINT_DTYPE:-bfloat16}}"
CERT_GLOBAL_MODE="${CERT_GLOBAL_MODE:-stream}"
CERT_TAU_RULE="${CERT_TAU_RULE:-global}"
CERT_HYBRID_MIN_LAYER_KEEP_RATIO="${CERT_HYBRID_MIN_LAYER_KEEP_RATIO:-0}"
STAGE="${MASK_GAP_STAGE:-full}"

mkdir -p "${OUT_DIR}"

echo "MASK_GAP_STAGE=${STAGE}"
echo "REPO_ROOT=${REPO_ROOT} OUT_DIR=${OUT_DIR}"

ARGS=(
  src/analysis/mask_score_gap_analysis.py
  --initial_model "${BASE_MODEL}"
  --final_model "${CKPT500_DIR}"
  --delta_log_dir "${DELTA_LOG_DIR}"
  --magnitude_milestones "${MAGNITUDE_MILESTONES}"
  --out_dir "${OUT_DIR}"
  --histogram_bins "${HISTOGRAM_BINS}"
  --sparsity_percent "${SPARSITY_PERCENT}"
  --cert_min_layer_keep_ratio "${CERT_MIN_LAYER_KEEP_RATIO}"
  --checkpoint_dtype "${CHECKPOINT_DTYPE}"
  --cert_global_mode "${CERT_GLOBAL_MODE}"
  --cert_tau_rule "${CERT_TAU_RULE}"
  --cert_hybrid_min_layer_keep_ratio "${CERT_HYBRID_MIN_LAYER_KEEP_RATIO}"
)
if [ "${CERT_MATCH_TIE_BREAK}" = "0" ] || [ "${CERT_MATCH_TIE_BREAK}" = "false" ]; then
  ARGS+=( --cert_no_match_tie_break )
fi
if [ "${FORCE_MAG_CACHE_REBUILD}" = "1" ]; then
  ARGS+=( --force_magnitude_cache_rebuild )
fi
if [ -n "${VERIFY_DELTAS_DIR}" ]; then
  ARGS+=( --verify_deltas_dir "${VERIFY_DELTAS_DIR}" )
fi
if [ -n "${RANDOM_SEEDS}" ]; then
  ARGS+=( --random_seeds "${RANDOM_SEEDS}" )
fi
if [ -n "${MAX_KEYS}" ]; then
  ARGS+=( --max_keys "${MAX_KEYS}" )
fi

if [ "${STAGE}" = "full" ]; then
  "${TRAIN_PY}" "${ARGS[@]}"
  "${TRAIN_PY}" scripts/report_mask_score_gap_plots.py --analysis-dir "${OUT_DIR}"
elif [ "${STAGE}" = "cache_only" ]; then
  ARGS+=( --execution_mode cache_only )
  "${TRAIN_PY}" "${ARGS[@]}"
elif [ "${STAGE}" = "baseline_shard" ]; then
  ARGS+=( --execution_mode baseline_shard )
  "${TRAIN_PY}" "${ARGS[@]}"
elif [ "${STAGE}" = "milestone_shard" ]; then
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: milestone_shard requires a Slurm array job (e.g. sbatch --array=0-3 ...)" >&2
    exit 1
  fi
  IFS=',' read -r -a MSA <<< "${MAGNITUDE_MILESTONES}"
  if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${#MSA[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for MAGNITUDE_MILESTONES (${#MSA[@]} steps)" >&2
    exit 1
  fi
  STEP="${MSA[$SLURM_ARRAY_TASK_ID]// /}"
  echo "milestone_shard step=${STEP} (array ${SLURM_ARRAY_TASK_ID})"
  ARGS+=( --execution_mode milestone_shard --milestone_shard_step "${STEP}" )
  "${TRAIN_PY}" "${ARGS[@]}"
elif [ "${STAGE}" = "merge_shards" ]; then
  ARGS+=( --execution_mode merge_shards )
  "${TRAIN_PY}" "${ARGS[@]}"
  "${TRAIN_PY}" scripts/report_mask_score_gap_plots.py --analysis-dir "${OUT_DIR}"
else
  echo "ERROR: Unknown MASK_GAP_STAGE=${STAGE}" >&2
  exit 1
fi

echo "Done stage ${STAGE}. OUT_DIR=${OUT_DIR}"

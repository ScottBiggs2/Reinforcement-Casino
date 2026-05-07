#!/bin/bash
# Full mask analysis suite vs a single ground-truth mask: Jaccard, CKA, per-layer CSV
# (sparsity + effective rank when enabled), JSON→CSV rollup, and plot_layer_metrics PNGs.
#
# Plot baselines (plot_layer_metrics_csv.py): theoretical only — no Monte Carlo.
#   Jaccard: E[J] and ±1σ from closed-form mean and Var (ρ = keep density, N = tensor length).
#   CKA null: E = 1/(N−1), σ² = 2/(N−1)² with N = total masked params (sum of n_params) unless overridden.
#
# Default paths match the Northeastern scratch layout; override with env vars.
#
# pipeline_common.sh defaults EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=1 (skip SVD / effective rank).
# This job resets that below so layer_metrics CSVs get effective_rank_* columns unless you opt out
# (see MASK_GT_SKIP_EFFECTIVE_RANK).
#
# Example (submit from repo root):
#   sbatch scripts/run_mask_analysis_vs_ground_truth.sh
#
# One-off overrides:
#   sbatch --export=ALL,MASK_ANALYSIS_DIR=/path/to/masks,GROUND_TRUTH_BASENAME=checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct.pt \
#     scripts/run_mask_analysis_vs_ground_truth.sh
#
# Fast run (skip effective rank, smaller CPU time):
#   sbatch --export=ALL,MASK_GT_SKIP_EFFECTIVE_RANK=1 scripts/run_mask_analysis_vs_ground_truth.sh
#
# Artifacts (all under MASK_ANALYSIS_DIR; override with env):
#   comparisons_vs_ground_truth/          — jaccard_*.json, cka_*.json, layer_metrics_*.csv, rollup CSVs
#   comparisons_vs_ground_truth/plots/    — *_plots.png
# Repo logs (submit cwd = REPO_ROOT):
#   logs/mask_gt_analysis_${SLURM_JOB_ID}.out|.err
#   logs/mask_gt_analysis_${RUN_ID}.log
#
# CKA vs effective rank (both enabled by default):
#   - CKA: GPU forward passes (mask_to_cka.py) → cka_*.json; not reused elsewhere.
#   - Effective rank: CPU linear-algebra on binary mask tensors (export_layer_metrics_csv.py); independent of CKA.
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --job-name=mask_gt_suite
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/mask_gt_analysis_%j.out
#SBATCH --error=logs/mask_gt_analysis_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"

# pipeline_common sets EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=1 by default. Mask-GT needs per-layer
# effective rank in CSVs for plots unless explicitly disabled (goes around that default).
if [ "${MASK_GT_SKIP_EFFECTIVE_RANK:-0}" = "1" ]; then
  export EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=1
else
  export EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=0
fi

# --- Analysis-specific defaults (override via sbatch --export or shell env) ---
MASK_ANALYSIS_DIR="${MASK_ANALYSIS_DIR:-/scratch/${USER:-unknown}/rl_casino_masks/<run_id>}"
GROUND_TRUTH_BASENAME="${GROUND_TRUTH_BASENAME:-checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct.pt}"

# RUN_ID only for log naming / provenance; outputs live under MASK_ANALYSIS_DIR.
export RUN_ID="${MASK_ANALYSIS_RUN_ID:-gt_analysis_$(basename "${MASK_ANALYSIS_DIR}")_${SLURM_JOB_ID:-local}}"
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-$RUN_ID}"

# Do not chain sparse DPO from this ad-hoc analysis job.
export PIPELINE_SKIP_SPARSE_LAUNCH="${PIPELINE_SKIP_SPARSE_LAUNCH:-1}"

# CKA + effective rank are part of the "full suite" for this job.
export RUN_MASK_CKA="${RUN_MASK_CKA:-1}"
# EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK is set immediately after sourcing pipeline_common (see above).

export PLOT_RANDOM_TRIALS="${PLOT_RANDOM_TRIALS:-1}"

# Skip complement / inverse masks when scanning (set to 1 to include them).
export MASK_ANALYSIS_SKIP_INVERSE="${MASK_ANALYSIS_SKIP_INVERSE:-1}"

# Write one random baseline mask matched to GT topology (same sparsity target as GT metadata).
export MASK_ANALYSIS_ENSURE_RANDOM_BASELINE="${MASK_ANALYSIS_ENSURE_RANDOM_BASELINE:-1}"

pipeline_setup

# Reduce CUDA fragmentation during repeated CKA runs (optional; harmless if unset).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GT_PATH="${MASK_ANALYSIS_DIR}/${GROUND_TRUTH_BASENAME}"
COMP_DIR="${MASK_ANALYSIS_DIR}/comparisons_vs_ground_truth"
PLOT_DIR="${COMP_DIR}/plots"
LOG_FILE="${REPO_ROOT}/logs/mask_gt_analysis_${RUN_ID}.log"
# Absolute paths for logs / provenance (Slurm .out/.err are relative to submit cwd = REPO_ROOT).
SLURM_OUT_ABS="${REPO_ROOT}/logs/mask_gt_analysis_${SLURM_JOB_ID:-local}.out"
SLURM_ERR_ABS="${REPO_ROOT}/logs/mask_gt_analysis_${SLURM_JOB_ID:-local}.err"

mkdir -p "$COMP_DIR" "$PLOT_DIR" logs

if [ ! -d "$MASK_ANALYSIS_DIR" ]; then
  echo "ERROR: MASK_ANALYSIS_DIR does not exist: ${MASK_ANALYSIS_DIR}" | tee -a "$LOG_FILE" >&2
  exit 1
fi
if [ ! -f "$GT_PATH" ]; then
  echo "ERROR: Ground-truth mask not found: ${GT_PATH}" | tee -a "$LOG_FILE" >&2
  exit 1
fi

echo "===== Mask analysis vs ground truth =====" | tee -a "$LOG_FILE"
echo "  REPO_ROOT=${REPO_ROOT}" | tee -a "$LOG_FILE"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-}" | tee -a "$LOG_FILE"
echo "  RUN_ID=${RUN_ID}" | tee -a "$LOG_FILE"
echo "  MASK_ANALYSIS_DIR=${MASK_ANALYSIS_DIR}" | tee -a "$LOG_FILE"
echo "  GROUND_TRUTH=${GT_PATH}" | tee -a "$LOG_FILE"
echo "  COMP_DIR=${COMP_DIR}" | tee -a "$LOG_FILE"
echo "  PLOT_DIR=${PLOT_DIR}" | tee -a "$LOG_FILE"
echo "  LOG_FILE=${LOG_FILE}" | tee -a "$LOG_FILE"
echo "  SLURM_STDOUT=${SLURM_OUT_ABS}" | tee -a "$LOG_FILE"
echo "  SLURM_STDERR=${SLURM_ERR_ABS}" | tee -a "$LOG_FILE"
echo "  RUN_MASK_CKA=${RUN_MASK_CKA}  EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=${EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK}" | tee -a "$LOG_FILE"

cmp_failures=0

run_one_cmp_step() {
  local desc="$1"
  shift
  {
    echo ""
    echo "---- ${desc} ----"
  } | tee -a "$LOG_FILE"
  set +e
  "$@"
  local ec=$?
  set -e
  if [ "$ec" -ne 0 ]; then
    echo "WARNING: ${desc} failed (exit ${ec})." | tee -a "$LOG_FILE" >&2
    return 1
  fi
  echo "OK: ${desc}" | tee -a "$LOG_FILE"
  return 0
}

# Same steps as run_jaccard_cka_export_pair in pipeline_common.sh, but with explicit paths.
run_jaccard_cka_export_pair() {
  local tag="$1" ma="$2" mb="$3"
  if [ ! -f "$ma" ] || [ ! -f "$mb" ]; then
    echo "SKIP comparison ${tag}: missing mask file(s)" | tee -a "$LOG_FILE"
    echo "  A=${ma}" | tee -a "$LOG_FILE"
    echo "  B=${mb}" | tee -a "$LOG_FILE"
    return 0
  fi

  run_one_cmp_step "Jaccard ${tag}" \
    timeout --signal=TERM --kill-after=60 "${MASK_COMPARISON_TIMEOUT_JACCARD}" \
      "$TRAIN_PY" src/cold_start/mask_to_jaccard.py "$ma" "$mb" \
        -o "${COMP_DIR}/jaccard_${tag}.json" || cmp_failures=$((cmp_failures + 1))

  # CKA JSON must exist for export_layer_metrics_csv to fill the `cka` column (plot_layer_metrics draws it).
  # Requires repo code with Tulu/DPO calibration (mask_to_cka + dpo_text_normalize). Pull latest before submit.
  if [ "${RUN_MASK_CKA}" = "1" ]; then
    run_one_cmp_step "CKA ${tag}" \
      timeout --signal=TERM --kill-after=60 "${MASK_COMPARISON_TIMEOUT_CKA}" \
        "$TRAIN_PY" src/cold_start/mask_to_cka.py "$ma" "$mb" \
          --model_name "$MODEL" \
          --dataset_name "$COLD_DATASET_HF" \
          --device cuda \
          --n_samples "${CKA_N_SAMPLES}" \
          --batch_size "${CKA_BATCH_SIZE}" \
          --seed 42 \
          -o "${COMP_DIR}/cka_${tag}.json" || cmp_failures=$((cmp_failures + 1))
  fi

  if [ -f "${COMP_DIR}/jaccard_${tag}.json" ]; then
    local -a export_cmd=(
      "$TRAIN_PY" src/cold_start/export_layer_metrics_csv.py "$ma" "$mb"
      --jaccard-json "${COMP_DIR}/jaccard_${tag}.json"
      -o "${COMP_DIR}/layer_metrics_${tag}.csv"
    )
    if [ -f "${COMP_DIR}/cka_${tag}.json" ]; then
      export_cmd+=( --cka-json "${COMP_DIR}/cka_${tag}.json" )
    fi
    if [ "${EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK:-0}" = "1" ]; then
      export_cmd+=( --skip_effective_rank )
    else
      export_cmd+=( --effective_rank_workers "${EXPORT_LAYER_METRICS_EFFECTIVE_RANK_WORKERS:-8}" )
    fi
    run_one_cmp_step "layer_metrics CSV ${tag}" "${export_cmd[@]}" || cmp_failures=$((cmp_failures + 1))
  fi
}

# Optional: random mask at GT sparsity (matched shapes) as an explicit empirical null row.
RANDOM_VS_GT="${MASK_ANALYSIS_DIR}/random_baseline_matched_to_ground_truth_seed${RANDOM_MASK_SEED:-42}.pt"
if [ "${MASK_ANALYSIS_ENSURE_RANDOM_BASELINE}" = "1" ]; then
  if [ ! -f "$RANDOM_VS_GT" ]; then
    rand_extra=()
    if [ -n "${MASK_RANDOM_SPARSITY_PERCENT:-}" ]; then
      rand_extra+=( --sparsity_percent "${MASK_RANDOM_SPARSITY_PERCENT}" )
    fi
    run_one_cmp_step "Generate random baseline mask matched to ground truth (${RANDOM_MASK_SEED:-42})" \
      "$TRAIN_PY" src/warm_start/random_mask_baseline.py \
        --reference_mask "$GT_PATH" \
        "${rand_extra[@]}" \
        --seed "${RANDOM_MASK_SEED:-42}" \
        --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
        --output_file "$RANDOM_VS_GT" || {
        echo "WARNING: could not create random baseline at ${RANDOM_VS_GT}" | tee -a "$LOG_FILE" >&2
      }
  else
    echo "Using existing random baseline: ${RANDOM_VS_GT}" | tee -a "$LOG_FILE"
  fi
fi

# --- Compare every top-level .pt in MASK_ANALYSIS_DIR to ground truth ---
shopt -s nullglob
mask_files=( "${MASK_ANALYSIS_DIR}"/*.pt )
shopt -u nullglob

if [ "${#mask_files[@]}" -eq 0 ]; then
  echo "ERROR: no .pt files in ${MASK_ANALYSIS_DIR}" | tee -a "$LOG_FILE" >&2
  exit 1
fi

for f in "${mask_files[@]}"; do
  base="$(basename "$f")"
  if [ "$base" = "$GROUND_TRUTH_BASENAME" ]; then
    echo "SKIP ground-truth file: ${base}" | tee -a "$LOG_FILE"
    continue
  fi
  if [ "${MASK_ANALYSIS_SKIP_INVERSE}" = "1" ] && [[ "$base" == *_inverse.pt ]]; then
    echo "SKIP inverse mask: ${base}" | tee -a "$LOG_FILE"
    continue
  fi

  stem="${base%.pt}"
  # Slug for filenames: alnum + underscore only, max length for sane paths.
  tag_raw="${stem//[^A-Za-z0-9._-]/_}"
  tag="gt_vs_${tag_raw}"
  if [ "${#tag}" -gt 180 ]; then
    tag="gt_vs_${tag_raw:0:170}"
  fi

  run_jaccard_cka_export_pair "$tag" "$f" "$GT_PATH"
done

run_one_cmp_step "convert_json_reports_to_csv" \
  "$TRAIN_PY" src/cold_start/convert_json_reports_to_csv.py \
    --input-dir "$COMP_DIR" --recursive || cmp_failures=$((cmp_failures + 1))

mkdir -p "$PLOT_DIR"
run_one_cmp_step "plot_layer_metrics_csv" \
  "$TRAIN_PY" src/cold_start/plot_layer_metrics_csv.py \
    --input-dir "$COMP_DIR" --recursive --pattern "layer_metrics_*.csv" \
    --output-dir "$PLOT_DIR" \
    --random-trials 1 \
    --random-seed 42 \
    --y-scale "${PLOT_Y_SCALE:-log}" \
    ${CKA_TOTAL_N:+--cka-total-n "${CKA_TOTAL_N}"} \
    --log-y-floor "${PLOT_LOG_Y_FLOOR:-1e-12}" \
    || cmp_failures=$((cmp_failures + 1))

shopt -s nullglob
_pngs=( "${PLOT_DIR}"/*.png )
shopt -u nullglob
if [ "${#_pngs[@]}" -eq 0 ]; then
  echo "ERROR: no PNGs under ${PLOT_DIR} after plot step (see plot_layer_metrics_csv output above)." | tee -a "$LOG_FILE" >&2
  cmp_failures=$((cmp_failures + 1))
fi

echo "Artifacts: JSON/CSV under ${COMP_DIR}; PNGs under ${PLOT_DIR}" | tee -a "$LOG_FILE"

manifest="${COMP_DIR}/RUN_MANIFEST_${RUN_ID}.txt"
{
  echo "mask_gt_suite run manifest"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "RUN_ID=${RUN_ID}"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "MASK_ANALYSIS_DIR=${MASK_ANALYSIS_DIR}"
  echo "GROUND_TRUTH=${GT_PATH}"
  echo "COMP_DIR=${COMP_DIR}"
  echo "PLOT_DIR=${PLOT_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "SLURM_STDOUT=${SLURM_OUT_ABS}"
  echo "SLURM_STDERR=${SLURM_ERR_ABS}"
  echo "RUN_MASK_CKA=${RUN_MASK_CKA}"
  echo "EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=${EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK}"
} > "$manifest"
echo "Wrote manifest: ${manifest}" | tee -a "$LOG_FILE"

if [ "${cmp_failures}" -gt 0 ]; then
  echo "WARNING: ${cmp_failures} step(s) reported failures. See ${LOG_FILE}" | tee -a "$LOG_FILE" >&2
  exit 1
fi
echo "Done." | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# Slurm job: N-mask interpretation suite (pairwise Jaccard JSON, optional CKA, layer_metrics CSV, rollup).
#
# Prepare a list file with one mask path per line (absolute paths recommended on cluster).
# Lines starting with # and blank lines are ignored. Example:
#
#   # masks/suite_dummy_test.txt
#   /scratch/$USER/masks/run_a/mask1.pt
#   /scratch/$USER/masks/run_b/mask2.pt
#
# Submit from repo root:
#   export MASK_SUITE_LIST_FILE=/scratch/$USER/masks/suite_dummy_test.txt
#   export MASK_SUITE_OUT_DIR=/scratch/$USER/mask_suite_out/run_${USER}_test1
#   sbatch scripts/sbatch_mask_interpretation_suite.sh
#
# Optional (sbatch --export=ALL,... or shell before sbatch):
#   MASK_SUITE_LABELS="dpo_a grpo_b fisher_c"   # same count as masks; default = file stems
#   MASK_SUITE_DEVICE=cpu                       # Jaccard / export only
#   MASK_SUITE_EXTENDED=both                    # none|param_bucket|decoder_layer|both
#   MASK_SUITE_SKIP_EFFECTIVE_RANK=1            # default 1 (fast); 0 for full SVD
#   MASK_SUITE_HEATMAP=1                        # jaccard_matrix.png (needs matplotlib)
#   MASK_SUITE_RUN_CKA=1                        # needs GPU job; set partition/GRES below
#   MASK_SUITE_CKA_MODEL=meta-llama/Llama-3.1-8B-Instruct
#   MASK_SUITE_CKA_DATASET=tulu3
#   MASK_SUITE_PROBE_REPORTS=1           # per-mask linear probes (GPU; uses same model as CKA)
#   MASK_SUITE_PROBE_MODE=grpo          # grpo|dpo for probe calibration
#   MASK_SUITE_RUN_PLOTS=1               # after suite: plot_layer_metrics_csv on pairwise/*.csv
#   MASK_SUITE_CKA_DEVICE=cuda
#   MASK_SUITE_CKA_N_SAMPLES=64
#   MASK_SUITE_CKA_BATCH_SIZE=4
#   MASK_SUITE_CKA_MAX_LENGTH=384          # shorter CKA forwards (speed)
#   MASK_SUITE_SUITE_FAST=1               # cap CKA/probe samples at 32 + skip effective rank
#   MASK_SUITE_NO_PROBE_PLOTS=1           # skip probe_plots/*.png (default: probe plots on with probes)
#   MASK_SUITE_PROBE_BUILTIN=all        # Irene corpora: all|none|syntax,semantics,math
#   MASK_SUITE_PROBE_BUILTIN_CV_FOLDS=3
#   MASK_SUITE_PROBE_BUILTIN_LAYER_STRIDE=1   # >1 subsamples MLP layers for builtin pass (speed)
#
# Default resources are CPU-only (Jaccard + CSV). For MASK_SUITE_RUN_CKA=1, use a GPU partition,
# e.g. sbatch --partition=gpu --gres=gpu:a100:1 --mem=128G --time=04:00:00 scripts/sbatch_mask_interpretation_suite.sh
# or edit the #SBATCH lines below to match your site.
#
# Smoke / debug (two random masks, different seeds, full bells): scripts/sbatch_mask_interpretation_suite_smoke.sh
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=mask_interp_suite
#SBATCH --output=logs/mask_interpretation_suite_%j.out
#SBATCH --error=logs/mask_interpretation_suite_%j.err

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
pipeline_setup

mkdir -p logs

MASK_SUITE_LIST_FILE="${MASK_SUITE_LIST_FILE:-}"
if [ -z "$MASK_SUITE_LIST_FILE" ] || [ ! -f "$MASK_SUITE_LIST_FILE" ]; then
  echo "ERROR: set MASK_SUITE_LIST_FILE to a readable text file (one mask .pt path per line)." >&2
  echo "  Example: export MASK_SUITE_LIST_FILE=/scratch/\${USER}/masks/my_suite.txt" >&2
  exit 2
fi

MASK_SUITE_OUT_DIR="${MASK_SUITE_OUT_DIR:-/scratch/${USER}/mask_interpretation_suite/run_${SLURM_JOB_ID:-local}}"
MASK_SUITE_DEVICE="${MASK_SUITE_DEVICE:-cpu}"
MASK_SUITE_EXTENDED="${MASK_SUITE_EXTENDED:-both}"
MASK_SUITE_SKIP_EFFECTIVE_RANK="${MASK_SUITE_SKIP_EFFECTIVE_RANK:-1}"
MASK_SUITE_HEATMAP="${MASK_SUITE_HEATMAP:-0}"
MASK_SUITE_RUN_CKA="${MASK_SUITE_RUN_CKA:-0}"
MASK_SUITE_CKA_MODEL="${MASK_SUITE_CKA_MODEL:-google/gemma-3-270m-it}"
MASK_SUITE_CKA_DATASET="${MASK_SUITE_CKA_DATASET:-tulu3}"
MASK_SUITE_CKA_DEVICE="${MASK_SUITE_CKA_DEVICE:-cuda}"
MASK_SUITE_CKA_N_SAMPLES="${MASK_SUITE_CKA_N_SAMPLES:-64}"
MASK_SUITE_CKA_BATCH_SIZE="${MASK_SUITE_CKA_BATCH_SIZE:-4}"

if [ "$MASK_SUITE_RUN_CKA" = "1" ]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

mapfile -t _raw_lines < "$MASK_SUITE_LIST_FILE"
paths=()
for line in "${_raw_lines[@]}"; do
  line="${line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [ -z "$line" ] && continue
  paths+=("$line")
done

if [ "${#paths[@]}" -lt 2 ]; then
  echo "ERROR: need at least two mask paths in $MASK_SUITE_LIST_FILE (got ${#paths[@]})." >&2
  exit 2
fi

for p in "${paths[@]}"; do
  if [ ! -f "$p" ]; then
    echo "ERROR: mask file not found: $p" >&2
    exit 2
  fi
done

echo "REPO_ROOT=$REPO_ROOT"
echo "LIST_FILE=$MASK_SUITE_LIST_FILE"
echo "OUT_DIR=$MASK_SUITE_OUT_DIR"
echo "N_MASKS=${#paths[@]}"
MASK_SUITE_CKA_MODEL="${MASK_SUITE_CKA_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MASK_SUITE_PROBE_REPORTS="${MASK_SUITE_PROBE_REPORTS:-0}"
MASK_SUITE_PROBE_MODE="${MASK_SUITE_PROBE_MODE:-grpo}"

echo "RUN_CKA=$MASK_SUITE_RUN_CKA DEVICE=$MASK_SUITE_DEVICE EXTENDED=$MASK_SUITE_EXTENDED"
echo "CKA_MODEL=$MASK_SUITE_CKA_MODEL PROBE_REPORTS=$MASK_SUITE_PROBE_REPORTS"

cmd=( "$TRAIN_PY" "${REPO_ROOT}/src/cold_start/mask_interpretation_suite.py" )
cmd+=( "${paths[@]}" )
cmd+=( --out-dir "$MASK_SUITE_OUT_DIR" )
cmd+=( --device "$MASK_SUITE_DEVICE" )
cmd+=( --extended-aggregates "$MASK_SUITE_EXTENDED" )
cmd+=( --cka-model "$MASK_SUITE_CKA_MODEL" )
cmd+=( --cka-dataset "$MASK_SUITE_CKA_DATASET" )
cmd+=( --cka-device "$MASK_SUITE_CKA_DEVICE" )
cmd+=( --cka-n-samples "$MASK_SUITE_CKA_N_SAMPLES" )
cmd+=( --cka-batch-size "$MASK_SUITE_CKA_BATCH_SIZE" )
cmd+=( --cka-max-length "${MASK_SUITE_CKA_MAX_LENGTH:-512}" )

if [ "${MASK_SUITE_SUITE_FAST:-0}" = "1" ]; then
  cmd+=( --suite-fast )
fi

if [ -n "${MASK_SUITE_LABELS:-}" ]; then
  # shellcheck disable=2206
  _labels=( $MASK_SUITE_LABELS )
  if [ "${#_labels[@]}" -ne "${#paths[@]}" ]; then
    echo "ERROR: MASK_SUITE_LABELS count (${#_labels[@]}) must equal mask count (${#paths[@]})." >&2
    exit 2
  fi
  cmd+=( --labels "${_labels[@]}" )
fi

if [ "$MASK_SUITE_SKIP_EFFECTIVE_RANK" = "1" ]; then
  cmd+=( --skip-effective-rank )
fi

if [ "$MASK_SUITE_HEATMAP" = "1" ]; then
  cmd+=( --heatmap )
fi

if [ "$MASK_SUITE_RUN_CKA" = "1" ]; then
  cmd+=( --run-cka )
fi

if [ "$MASK_SUITE_PROBE_REPORTS" = "1" ]; then
  cmd+=( --probe-reports --probe-mode "$MASK_SUITE_PROBE_MODE" )
  if [ -n "${MASK_SUITE_PROBE_DATASET:-}" ]; then
    cmd+=( --probe-dataset "$MASK_SUITE_PROBE_DATASET" )
  fi
  cmd+=( --probe-device "${MASK_SUITE_PROBE_DEVICE:-cuda}" )
  cmd+=( --probe-n-samples "${MASK_SUITE_PROBE_N_SAMPLES:-64}" )
  cmd+=( --probe-batch-size "${MASK_SUITE_PROBE_BATCH_SIZE:-4}" )
  cmd+=( --probe-max-length "${MASK_SUITE_PROBE_MAX_LENGTH:-512}" )
  cmd+=( --probe-builtin-datasets "${MASK_SUITE_PROBE_BUILTIN:-all}" )
  cmd+=( --probe-builtin-cv-folds "${MASK_SUITE_PROBE_BUILTIN_CV_FOLDS:-3}" )
  cmd+=( --probe-builtin-layer-stride "${MASK_SUITE_PROBE_BUILTIN_LAYER_STRIDE:-1}" )
  if [ "${MASK_SUITE_NO_PROBE_PLOTS:-0}" = "1" ]; then
    cmd+=( --no-probe-plots )
  fi
fi

echo "RUN: ${cmd[*]}"
"${cmd[@]}"

if [ "${MASK_SUITE_RUN_PLOTS:-0}" = "1" ]; then
  PDIR="${MASK_SUITE_OUT_DIR}/plots"
  mkdir -p "$PDIR"
  echo "=== plot_layer_metrics_csv -> $PDIR ==="
  "$TRAIN_PY" "${REPO_ROOT}/src/cold_start/plot_layer_metrics_csv.py" \
    --input-dir "${MASK_SUITE_OUT_DIR}/pairwise" \
    --pattern "layer_metrics_*.csv" \
    --output-dir "$PDIR" \
    --jaccard-mc-trials "${MASK_SUITE_PLOT_MC_TRIALS:-200}" \
    --jaccard-mc-seed 42 \
    --y-scale "${MASK_SUITE_PLOT_Y_SCALE:-linear}" \
    ${CKA_TOTAL_N:+--cka-total-n "${CKA_TOTAL_N}"} \
    || echo "WARNING: plot_layer_metrics_csv exited non-zero" >&2
fi

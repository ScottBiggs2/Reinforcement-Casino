#!/bin/bash
# Standalone Slurm job: (re)run cold-start mask generation only (Fisher + CAV/SNIP),
# skipping any outputs that already exist for a given RUN_ID.
#
# Usage (repo root):
#   sbatch --export=ALL,PIPELINE_RUN_ID=<RUN_ID>,RUN_ID=<RUN_ID> scripts/run_cold_masks_only_slurm.sh
#
# Optional env:
#   MODEL, COLD_DATASET_HF, SPARSITY_LIST, COLD_FISHER_N_CALIB, COLD_CAV_SUBSET, COLD_CAV_NUM_BATCHES,
#   MIN_LAYER_KEEP_RATIO, MASK_TIMEOUT
#
# Notes:
# - Uses the training env (TRAIN_PY) from scripts/pipeline_common.sh
# - Writes into ${MASK_OUT_BASE}/${RUN_ID} with the same filenames the pipeline expects.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --job-name=cold_masks
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/cold_masks_%j.out
#SBATCH --error=logs/cold_masks_%j.err

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

echo "===== Cold masks only (${RUN_ID}) ====="
echo "Model: ${MODEL}"
echo "Dataset (cold): ${COLD_DATASET_HF}"
echo "Sparsities: ${SPARSITY_LIST[*]}"

mask_base="${MASK_OUT_BASE}/${RUN_ID}"
mkdir -p "$mask_base" logs

# Derive the same sanitized model name used in pipeline filenames.
ds="${DPO_DATASETS[0]}"
_p=()
while IFS= read -r line; do _p+=("$line"); done < <(pipeline_dpo_path_components "$MODEL" "$ds")
model_sanitized="${_p[0]}"

run_one() {
  local desc="$1"
  shift
  echo ""
  echo "---- ${desc} ----"
  set +e
  timeout --signal=TERM --kill-after=60 "${MASK_TIMEOUT}" "$@"
  local ec=$?
  set -e
  if [ "$ec" -ne 0 ]; then
    echo "WARNING: ${desc} failed (exit ${ec})." >&2
    return 1
  fi
  echo "OK: ${desc}"
  return 0
}

for sparsity in "${SPARSITY_LIST[@]}"; do
  # These filenames match scripts/pipeline_common.sh
  fisher_out="${mask_base}/cold_fisher_${model_sanitized}_sparsity${sparsity}pct_n${COLD_FISHER_N_CALIB}.pt"
  cav_out="${mask_base}/cold_cav_${model_sanitized}_sparsity${sparsity}pct.pt"

  if [ -f "$fisher_out" ]; then
    echo "SKIP cold Fisher (exists): ${fisher_out}"
  else
    run_one "cold Fisher sparsity=${sparsity}" \
      "$TRAIN_PY" src/cold_start/cold_mask_finder.py \
        --model_name "$MODEL" \
        --dataset_name "$COLD_DATASET_HF" \
        --sparsity_percent "$sparsity" \
        --n_calibration_samples "${COLD_FISHER_N_CALIB}" \
        --mini_batch_size 4 \
        --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
        --output_file "$fisher_out" || true
  fi

  if [ -f "$cav_out" ]; then
    echo "SKIP cold CAV (exists): ${cav_out}"
  else
    run_one "cold CAV sparsity=${sparsity}" \
      "$TRAIN_PY" src/cold_start/cav_cold_mask_finder.py \
        --model_name "$MODEL" \
        --dataset_name "$COLD_DATASET_HF" \
        --method cav \
        --sparsity_percent "$sparsity" \
        --subset_size "${COLD_CAV_SUBSET}" \
        --num_batches "${COLD_CAV_NUM_BATCHES}" \
        --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
        --output_file "$cav_out" || true
  fi
done

echo "===== Cold masks only complete (${RUN_ID}) ====="


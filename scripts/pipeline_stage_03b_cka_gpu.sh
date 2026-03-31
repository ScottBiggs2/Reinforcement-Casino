#!/bin/bash
# Stage 3b (GPU-only): activation CKA for mask comparisons.
# Runs only when RUN_MASK_CKA=1.
#
# Produces cka_*.json next to existing jaccard_*.json and refreshes layer_metrics_*.csv
# to include the CKA column (by re-running export_layer_metrics_csv.py with --cka-json).
#
# This is split out from stage 3a (CPU) to avoid GPU-idle cancellation policies on clusters
# when the non-CKA comparison work is CPU-heavy.
#
# Required env: RUN_ID / PIPELINE_RUN_ID already set by the pipeline chain.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=pipe_p3b_cka
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p3b_cka.out
#SBATCH --error=logs/pipeline_%j_p3b_cka.err

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

echo "===== STAGE 3b (GPU): mask CKA (${RUN_ID}) ====="
if [ "${RUN_MASK_CKA:-0}" != "1" ]; then
  echo "RUN_MASK_CKA!=1; nothing to do. Proceeding to next stage."
  pipeline_submit_next_stage "pipeline_stage_04_sparse.sh"
  exit 0
fi

mask_dir="${MASK_OUT_BASE}/${RUN_ID}"
comp_dir="${mask_dir}/comparisons"
mkdir -p "$comp_dir" logs

ds="${DPO_DATASETS[0]}"
_p=()
while IFS= read -r line; do _p+=("$line"); done < <(pipeline_dpo_path_components "$MODEL" "$ds")
model_sanitized="${_p[0]}"
ds_sanitized="${_p[1]}"

sp_list=("${SPARSITY_LIST[@]}")

run_one() {
  local desc="$1"
  shift
  echo ""
  echo "---- ${desc} ----"
  set +e
  timeout --signal=TERM --kill-after=60 "${MASK_COMPARISON_TIMEOUT_CKA}" "$@"
  local ec=$?
  set -e
  if [ "$ec" -ne 0 ]; then
    echo "WARNING: ${desc} failed (exit ${ec}). Continuing." >&2
    return 1
  fi
  echo "OK: ${desc}"
  return 0
}

for sparsity in "${sp_list[@]}"; do
  sp_safe=$(echo "${sparsity}" | tr '.' '_')

  warm_mag="${mask_dir}/warm_magnitude_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step${TARGET_STEP_DPO}.pt"
  warm_mom="${mask_dir}/warm_momentum_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step${TARGET_STEP_DPO}.pt"
  warm_fish="${mask_dir}/warm_fisher_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step${TARGET_STEP_DPO}.pt"
  cold_fish="${mask_dir}/cold_fisher_${model_sanitized}_sparsity${sparsity}pct_n${COLD_FISHER_N_CALIB}.pt"
  cold_cav="${mask_dir}/cold_cav_${model_sanitized}_sparsity${sparsity}pct.pt"

  # Match stage-3 tags so downstream tools can find the files.
  tags=( "sp${sp_safe}_wm_vs_cf" "sp${sp_safe}_wmom_vs_cc" "sp${sp_safe}_wf_vs_cf" )
  mas=( "$warm_mag" "$warm_mom" "$warm_fish" )
  mbs=( "$cold_fish" "$cold_cav" "$cold_fish" )

  for i in 0 1 2; do
    tag="${tags[$i]}"
    ma="${mas[$i]}"
    mb="${mbs[$i]}"
    jjson="${comp_dir}/jaccard_${tag}.json"
    cjson="${comp_dir}/cka_${tag}.json"
    csv="${comp_dir}/layer_metrics_${tag}.csv"

    if [ ! -f "$ma" ] || [ ! -f "$mb" ]; then
      echo "SKIP CKA ${tag}: missing mask file(s)"
      continue
    fi

    run_one "CKA ${tag}" \
      "$TRAIN_PY" src/cold_start/mask_to_cka.py "$ma" "$mb" \
        --model_name "$MODEL" \
        --dataset_name "$COLD_DATASET_HF" \
        --device cuda \
        --n_samples "${CKA_N_SAMPLES}" \
        --batch_size "${CKA_BATCH_SIZE}" \
        --seed 42 \
        -o "$cjson" || true

    # If we have a Jaccard JSON from stage 3a, refresh the CSV to include CKA.
    if [ -f "$jjson" ] && [ -f "$cjson" ]; then
      cmd=( "$TRAIN_PY" src/cold_start/export_layer_metrics_csv.py "$ma" "$mb" --jaccard-json "$jjson" --cka-json "$cjson" -o "$csv" )
      if [ "${EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK:-1}" = "1" ]; then
        cmd+=( --skip_effective_rank )
      else
        cmd+=( --effective_rank_workers "${EXPORT_LAYER_METRICS_EFFECTIVE_RANK_WORKERS:-4}" )
      fi
      run_one "refresh layer_metrics CSV (with CKA) ${tag}" "${cmd[@]}" || true
    fi
  done
done

echo "CKA artifacts under: ${comp_dir}"
pipeline_submit_next_stage "pipeline_stage_04_sparse.sh"


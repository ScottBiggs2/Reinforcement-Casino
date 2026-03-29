# shellcheck shell=bash
# Shared RL Casino full-pipeline logic (sourced by run_full_pipeline.sh and pipeline_stage_*.sh).
# For 8h cluster caps, use: bash scripts/submit_pipeline_chain.sh
set -euo pipefail

########################################
# 0. Paths and global config
########################################

# Slurm copies batch scripts to /var/spool/slurmd/... — BASH_SOURCE is NOT the repo path.
# Always submit from the repo root:  cd .../rl_casino && sbatch scripts/run_full_pipeline.sh
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"

if [ -n "${PIPELINE_RUN_ID:-}" ]; then
  export RUN_ID="${PIPELINE_RUN_ID}"
elif [ -n "${RUN_ID_OVERRIDE:-}" ]; then
  export RUN_ID="${RUN_ID_OVERRIDE}"
else
  export RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
fi
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-$RUN_ID}"
echo "Full pipeline RUN_ID=${RUN_ID}"
echo "Repo root: ${REPO_ROOT}"

# Training environment (must already exist and have requirements installed)
TRAIN_ENV="/scratch/biggs.s/conda_envs/rl_casino"
TRAIN_PY="${TRAIN_ENV}/bin/python"

# Eval environment (for run_evals_slurm / verify_coding, separate env)
EVAL_ENV="/scratch/biggs.s/conda_envs/rl_casino_eval"

# Scratch / output roots
TRAIN_OUT_BASE="/scratch/biggs.s/rl_casino_train"
MASK_OUT_BASE="/scratch/biggs.s/rl_casino_masks"
SPARSE_OUT_BASE="/scratch/biggs.s/rl_casino_sparse_train"
EVAL_OUT_BASE="/scratch/biggs.s/rl_casino_eval_runs"

mkdir -p "$TRAIN_OUT_BASE" "$MASK_OUT_BASE" "$SPARSE_OUT_BASE" "$EVAL_OUT_BASE" logs

# Model / dataset (export HF_TOKEN for gated Llama; Tulu3 = allenai/llama-3.1-tulu-3-8b-preference-mixture)
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DPO_DATASETS=("tulu3")       # dataset registry keys; drives cold masks + sparse DPO
NUM_STEPS_DPO=500
# SUBSET_DPO unset or empty = full dataset (omit --subset_size). Example: SUBSET_DPO=4096 for a cap
SUBSET_DPO="${SUBSET_DPO:-}"

SPARSITY_LIST=("97.5")
# Must match an existing deltas_step_<N>.pt (warm masks). With 500 steps & defaults: deltas at 50..min(50,500/10)=50 → [50]; if DELTA_LOG_END_STEP=200, checkpoints include 200 while training runs 500 steps.
TARGET_STEP_DPO="${TARGET_STEP_DPO:-200}"
MIN_LAYER_KEEP_RATIO="0.0025"

# DPO_train delta schedule: every DELTA_LOG_INTERVAL steps up to DELTA_LOG_END_STEP (omit for auto = 10%% of num_steps)
DELTA_LOG_INTERVAL="${DELTA_LOG_INTERVAL:-50}"
DELTA_LOG_END_STEP="${DELTA_LOG_END_STEP:-200}"

# Eval: empty EVAL_LIMIT = full benchmarks (no --limit). EVAL_FORCE_HF_BACKEND=0 → vLLM when installed (full throughput)
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_FORCE_HF_BACKEND="${EVAL_FORCE_HF_BACKEND:-0}"

# Cold-start calibration (override for ablations; filenames include Fisher N)
COLD_FISHER_N_CALIB="${COLD_FISHER_N_CALIB:-256}"
COLD_CAV_SUBSET="${COLD_CAV_SUBSET:-256}"
COLD_CAV_NUM_BATCHES="${COLD_CAV_NUM_BATCHES:-16}"

# Mask comparison stage: set RUN_MASK_CKA=1 for GPU-heavy activation CKA (add time budget)
RUN_MASK_CKA="${RUN_MASK_CKA:-0}"
MASK_COMPARISON_TIMEOUT_JACCARD="${MASK_COMPARISON_TIMEOUT_JACCARD:-600}"
MASK_COMPARISON_TIMEOUT_CKA="${MASK_COMPARISON_TIMEOUT_CKA:-$((2 * 60 * 60))}"
CKA_N_SAMPLES="${CKA_N_SAMPLES:-64}"
CKA_BATCH_SIZE="${CKA_BATCH_SIZE:-2}"
PLOT_RANDOM_TRIALS="${PLOT_RANDOM_TRIALS:-3}"

# Parallel sparse jobs (pipeline_stage_04): per-GPU-job wall time; eval dependency on all sparse jobs
SPARSE_SLURM_TIME="${SPARSE_SLURM_TIME:-24:00:00}"
# afterok = run evals only if every sparse job succeeded; afterany = run evals when all have finished (any exit code)
PIPELINE_SPARSE_EVAL_DEPENDENCY="${PIPELINE_SPARSE_EVAL_DEPENDENCY:-afterok}"

# Timeouts (seconds) — default to ~8h wall per Slurm job (see submit_pipeline_chain.sh).
# Override when running a single long-horizon sbatch (e.g. TRAIN_TIMEOUT_PER_DATASET=$((16*60*60))).
TRAIN_TIMEOUT_PER_DATASET="${TRAIN_TIMEOUT_PER_DATASET:-$((7 * 60 * 60 + 30 * 60))}"   # 7h30m
MASK_TIMEOUT="${MASK_TIMEOUT:-$((7 * 60 * 60 + 30 * 60))}"
SPARSE_TIMEOUT_PER_MASK="${SPARSE_TIMEOUT_PER_MASK:-$((7 * 60 * 60 + 30 * 60))}"

# Per-job soft cap (stay under typical Slurm 8h wall)
GLOBAL_MAX_SECONDS="${GLOBAL_MAX_SECONDS:-$((7 * 60 * 60 + 45 * 60))}"
START_TS=$(date +%s)

check_budget() {
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - START_TS))
  if (( elapsed > GLOBAL_MAX_SECONDS )); then
    echo "GLOBAL TIMEOUT: elapsed ${elapsed}s > ${GLOBAL_MAX_SECONDS}s" >&2
    exit 1
  fi
}

# Must match src/full_training/DPO_train.py (sanitize_model_name) + dataset_registry (sanitized_name).
pipeline_dpo_path_components() {
  PIPELINE_MODEL="$1" PIPELINE_DS_KEY="$2" PIPELINE_REPO="$REPO_ROOT" "$TRAIN_PY" -c "
import os, sys
sys.path.insert(0, os.environ['PIPELINE_REPO'])
from src.utils.dataset_registry import get_dataset_config

def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace('/', '_').replace('-', '_').lower()
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized.strip('_')

m = os.environ['PIPELINE_MODEL']
k = os.environ['PIPELINE_DS_KEY']
cfg = get_dataset_config(k)
ms = sanitize_model_name(m)
ds = cfg['sanitized_name']
print(ms)
print(ds)
print(f'{ms}_{ds}')
"
}

########################################
# 1. Env setup (call once per Slurm job)
########################################

pipeline_setup() {
  START_TS=$(date +%s)
  echo "Using training env: ${TRAIN_ENV}"
  if [ ! -x "$TRAIN_PY" ]; then
    echo "ERROR: TRAIN_PY not found at ${TRAIN_PY}" >&2
    exit 1
  fi
  export PATH="${TRAIN_ENV}/bin:${PATH}"

  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export RL_CASINO_WARM_MASK_SCORE_DEVICE="${RL_CASINO_WARM_MASK_SCORE_DEVICE:-cpu}"
  export RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL="${RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL:-250000000}"

  echo "Mask runtime knobs:"
  echo "  PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
  echo "  RL_CASINO_WARM_MASK_SCORE_DEVICE=${RL_CASINO_WARM_MASK_SCORE_DEVICE}"
  echo "  RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL=${RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL}"

  echo "Environment check (training):"
  "$TRAIN_PY" -c "import torch, trl; print(f'Torch: {torch.__version__}, TRL: {trl.__version__}')" || {
    echo "WARNING: Could not import torch+trl in training env; ensure requirements are installed." >&2
  }

  if [ -z "${COLD_DATASET_HF:-}" ]; then
    COLD_DATASET_HF=$(PIPELINE_DS_KEY="${DPO_DATASETS[0]}" PIPELINE_REPO="$REPO_ROOT" "$TRAIN_PY" -c "
import os, sys
sys.path.insert(0, os.environ['PIPELINE_REPO'])
from src.utils.dataset_registry import get_dataset_config
print(get_dataset_config(os.environ['PIPELINE_DS_KEY'])['hf_id'])
")
  fi
  echo "Cold-start / CKA calibration dataset (HF): ${COLD_DATASET_HF}"
}

# Submit the next pipeline stage after this job finishes successfully (Slurm dependency chain).
pipeline_submit_next_stage() {
  local next_script="$1"
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: pipeline_submit_next_stage needs SLURM_JOB_ID (run from an sbatch job)" >&2
    exit 1
  fi
  local jid
  jid=$(sbatch --parsable \
    --dependency=afterok:"${SLURM_JOB_ID}" \
    --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
    "${REPO_ROOT}/scripts/${next_script}")
  echo "Chained next stage: ${next_script} → Slurm job ${jid} (afterok:${SLURM_JOB_ID})"
}

########################################
# 2. Dense DPO training
########################################

run_dense_dpo() {
  local ds
  for ds in "${DPO_DATASETS[@]}"; do
    check_budget
    echo "=== Dense DPO training for dataset=${ds} ==="

    local out_base cache_dir run_name
    out_base="${TRAIN_OUT_BASE}/${RUN_ID}"
    cache_dir="/scratch/biggs.s/hf_cache/datasets"
    run_name="fullpipe_dpo_${ds}_${RUN_ID}"

    mkdir -p "$out_base"

    local subset_args=()
    if [ -n "${SUBSET_DPO:-}" ]; then
      subset_args+=(--subset_size "$SUBSET_DPO")
    fi
    local delta_end_args=()
    if [ -n "${DELTA_LOG_END_STEP:-}" ]; then
      delta_end_args+=(--delta_log_end_step "$DELTA_LOG_END_STEP")
    fi

    timeout --signal=TERM --kill-after=60 "${TRAIN_TIMEOUT_PER_DATASET}" \
      "$TRAIN_PY" src/full_training/DPO_train.py \
        --model_name "$MODEL" \
        --dataset "$ds" \
        --num_steps "$NUM_STEPS_DPO" \
        "${subset_args[@]}" \
        --delta_log_interval "${DELTA_LOG_INTERVAL}" \
        "${delta_end_args[@]}" \
        --output_base_dir "$out_base" \
        --dataset_cache_dir "$cache_dir" \
        --use_wandb \
        --run_name "$run_name" 2>&1 | tee "logs/full_pipeline_dpo_${ds}_${RUN_ID}.log"

    local rc=$?
    if (( rc != 0 )); then
      echo "ERROR: Dense DPO training failed for ${ds} (exit ${rc}). Aborting pipeline." >&2
      exit 1
    fi
  done
}

########################################
# 3. Mask building (warm + cold)
########################################

run_masks() {
  check_budget
  echo "=== Mask building (warm + cold) ==="

  # For now, assume single dataset from DPO_DATASETS[0]
  local ds="${DPO_DATASETS[0]}"

  local model_sanitized ds_sanitized subdir delta_dir mask_base mask_failures log_file
  _p=()
  while IFS= read -r line; do _p+=("$line"); done < <(pipeline_dpo_path_components "$MODEL" "$ds")
  model_sanitized="${_p[0]}"
  ds_sanitized="${_p[1]}"
  subdir="${_p[2]}"
  delta_dir="${TRAIN_OUT_BASE}/${RUN_ID}/deltas/${subdir}"
  mask_base="${MASK_OUT_BASE}/${RUN_ID}"
  log_file="logs/full_pipeline_masks_${RUN_ID}.log"
  mask_failures=0

  echo "Using DPO delta dir: ${delta_dir}"
  mkdir -p "$mask_base"

  # Run each mask step independently: OOM or other errors log a warning and continue.
  # (Previously a single failure under set -e aborted the whole mask stage and the pipeline.)
  run_one_mask_step() {
    local desc="$1"
    shift
    {
      echo ""
      echo "---- ${desc} ----"
    } | tee -a "$log_file"
    set +e
    timeout --signal=TERM --kill-after=60 "${MASK_TIMEOUT}" "$@"
    local ec=$?
    set -e
    if [ "$ec" -ne 0 ]; then
      echo "WARNING: ${desc} failed (exit ${ec}). Continuing with remaining mask steps." | tee -a "$log_file" >&2
      mask_failures=$((mask_failures + 1))
    else
      echo "OK: ${desc}" | tee -a "$log_file"
    fi
  }

  local method sparsity

  echo "Warm-start masks from ${delta_dir}"
  for method in magnitude momentum fisher; do
    for sparsity in "${SPARSITY_LIST[@]}"; do
      run_one_mask_step "warm ${method} sparsity=${sparsity}" \
        "$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
          --delta_log_dir "$delta_dir" \
          --method "$method" \
          --sparsity_percent "$sparsity" \
          --target_step "${TARGET_STEP_DPO}" \
          --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
          --output_file "${mask_base}/warm_${method}_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step${TARGET_STEP_DPO}.pt"
    done
  done

  echo "Cold-start Fisher masks"
  for sparsity in "${SPARSITY_LIST[@]}"; do
    run_one_mask_step "cold Fisher sparsity=${sparsity}" \
      "$TRAIN_PY" src/cold_start/cold_mask_finder.py \
        --model_name "$MODEL" \
        --dataset_name "$COLD_DATASET_HF" \
        --sparsity_percent "$sparsity" \
        --n_calibration_samples "${COLD_FISHER_N_CALIB}" \
        --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
        --output_file "${mask_base}/cold_fisher_${model_sanitized}_sparsity${sparsity}pct_n${COLD_FISHER_N_CALIB}.pt"
  done

  echo "Cold-start CAV masks"
  for sparsity in "${SPARSITY_LIST[@]}"; do
    run_one_mask_step "cold CAV sparsity=${sparsity}" \
      "$TRAIN_PY" src/cold_start/cav_cold_mask_finder.py \
        --model_name "$MODEL" \
        --dataset_name "$COLD_DATASET_HF" \
        --method cav \
        --sparsity_percent "$sparsity" \
        --subset_size "${COLD_CAV_SUBSET}" \
        --num_batches "${COLD_CAV_NUM_BATCHES}" \
        --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
        --output_file "${mask_base}/cold_cav_${model_sanitized}_sparsity${sparsity}pct.pt"
  done

  if [ "${mask_failures}" -gt 0 ]; then
    echo "WARNING: ${mask_failures} mask step(s) failed in total. Pipeline continues; sparse training uses any .pt files under ${mask_base}." | tee -a "$log_file" >&2
  else
    echo "All mask steps completed successfully." | tee -a "$log_file"
  fi
}

########################################
# 3b. Mask comparison (Jaccard, optional CKA, CSV, plots)
########################################

run_mask_comparisons() {
  check_budget
  echo "=== Mask comparison (Jaccard, optional CKA, CSV, plots) ==="

  local ds="${DPO_DATASETS[0]}"
  local _p=()
  while IFS= read -r line; do _p+=("$line"); done < <(pipeline_dpo_path_components "$MODEL" "$ds")
  local model_sanitized="${_p[0]}"
  local ds_sanitized="${_p[1]}"

  local mask_dir="${MASK_OUT_BASE}/${RUN_ID}"
  local comp_dir="${mask_dir}/comparisons"
  local log_file="logs/full_pipeline_mask_comparisons_${RUN_ID}.log"
  local cmp_failures=0
  mkdir -p "$comp_dir"

  run_one_cmp_step() {
    local desc="$1"
    shift
    {
      echo ""
      echo "---- ${desc} ----"
    } | tee -a "$log_file"
    set +e
    "$@"
    local ec=$?
    set -e
    if [ "$ec" -ne 0 ]; then
      echo "WARNING: ${desc} failed (exit ${ec})." | tee -a "$log_file" >&2
      return 1
    fi
    echo "OK: ${desc}" | tee -a "$log_file"
    return 0
  }

  local sparsity sp_safe warm_mag warm_mom warm_fish cold_fish cold_cav
  local tags mas mbs i tag ma mb
  local -a export_cmd

  for sparsity in "${SPARSITY_LIST[@]}"; do
    sp_safe=$(echo "${sparsity}" | tr '.' '_')

    warm_mag="${mask_dir}/warm_magnitude_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step${TARGET_STEP_DPO}.pt"
    warm_mom="${mask_dir}/warm_momentum_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step${TARGET_STEP_DPO}.pt"
    warm_fish="${mask_dir}/warm_fisher_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step${TARGET_STEP_DPO}.pt"
    cold_fish="${mask_dir}/cold_fisher_${model_sanitized}_sparsity${sparsity}pct_n${COLD_FISHER_N_CALIB}.pt"
    cold_cav="${mask_dir}/cold_cav_${model_sanitized}_sparsity${sparsity}pct.pt"

    tags=( "sp${sp_safe}_wm_vs_cf" "sp${sp_safe}_wmom_vs_cc" "sp${sp_safe}_wf_vs_cf" )
    mas=( "$warm_mag" "$warm_mom" "$warm_fish" )
    mbs=( "$cold_fish" "$cold_cav" "$cold_fish" )

    for i in 0 1 2; do
      tag="${tags[$i]}"
      ma="${mas[$i]}"
      mb="${mbs[$i]}"
      if [ ! -f "$ma" ] || [ ! -f "$mb" ]; then
        echo "SKIP comparison ${tag}: missing mask file(s)" | tee -a "$log_file"
        echo "  A=${ma}" | tee -a "$log_file"
        echo "  B=${mb}" | tee -a "$log_file"
        continue
      fi

      run_one_cmp_step "Jaccard ${tag}" \
        timeout --signal=TERM --kill-after=60 "${MASK_COMPARISON_TIMEOUT_JACCARD}" \
          "$TRAIN_PY" src/cold_start/mask_to_jaccard.py "$ma" "$mb" \
            -o "${comp_dir}/jaccard_${tag}.json" || cmp_failures=$((cmp_failures + 1))

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
              -o "${comp_dir}/cka_${tag}.json" || cmp_failures=$((cmp_failures + 1))
      fi

      if [ -f "${comp_dir}/jaccard_${tag}.json" ]; then
        export_cmd=(
          "$TRAIN_PY" src/cold_start/export_layer_metrics_csv.py "$ma" "$mb"
          --jaccard-json "${comp_dir}/jaccard_${tag}.json"
          -o "${comp_dir}/layer_metrics_${tag}.csv"
        )
        if [ -f "${comp_dir}/cka_${tag}.json" ]; then
          export_cmd+=( --cka-json "${comp_dir}/cka_${tag}.json" )
        fi
        run_one_cmp_step "layer_metrics CSV ${tag}" "${export_cmd[@]}" || cmp_failures=$((cmp_failures + 1))
      fi
    done
  done

  run_one_cmp_step "convert_json_reports_to_csv" \
    "$TRAIN_PY" src/cold_start/convert_json_reports_to_csv.py \
      --input-dir "$comp_dir" --recursive || cmp_failures=$((cmp_failures + 1))

  run_one_cmp_step "plot_layer_metrics_csv" \
    "$TRAIN_PY" src/cold_start/plot_layer_metrics_csv.py \
      --input-dir "$comp_dir" --recursive --pattern "layer_metrics_*.csv" \
      --random-trials "${PLOT_RANDOM_TRIALS}" || cmp_failures=$((cmp_failures + 1))

  echo "Mask comparison artifacts under: ${comp_dir}"
  if [ "${cmp_failures}" -gt 0 ]; then
    echo "WARNING: ${cmp_failures} mask comparison step(s) had failures. See ${log_file}" | tee -a "$log_file" >&2
  fi
}

########################################
# 4. Sparse DPO training
########################################

# One mask: NUM_STEPS_DPO on DPO_DATASETS[0] (e.g. tulu3), same as dense pipeline.
run_sparse_dpo_one_mask() {
  local mask="$1"
  check_budget
  local mask_name
  mask_name=$(basename "$mask")
  echo "-> Sparse DPO run for mask=${mask_name}"

  local out_base cache_dir run_name
  out_base="${SPARSE_OUT_BASE}/${RUN_ID}/${mask_name%.pt}"
  cache_dir="/scratch/biggs.s/hf_cache/datasets"
  run_name="fullpipe_sparse_${mask_name%.*}_${RUN_ID}"

  mkdir -p "$out_base"

  local sparse_subset_args=()
  if [ -n "${SUBSET_DPO:-}" ]; then
    sparse_subset_args+=(--subset_size "$SUBSET_DPO")
  fi

  timeout --signal=TERM --kill-after=60 "${SPARSE_TIMEOUT_PER_MASK}" \
    "$TRAIN_PY" src/full_training/sparse_dpo_efficiency.py \
      --model_name "$MODEL" \
      --checkpoint None \
      --mask "$mask" \
      --dataset "${DPO_DATASETS[0]}" \
      --n_steps "$NUM_STEPS_DPO" \
      --batch_size 4 \
      --grad_accum 4 \
      "${sparse_subset_args[@]}" \
      --optimizer sparse_adamw \
      --block_size 32 \
      --use_wandb \
      --save_csv \
      --output_base_dir "$out_base" \
      --dataset_cache_dir "$cache_dir" \
      --run_name "$run_name" 2>&1 | tee "logs/full_pipeline_sparse_${mask_name%.*}_${RUN_ID}.log"

  local rc=$?
  if (( rc != 0 )); then
    echo "WARNING: Sparse DPO failed for mask ${mask_name} (exit ${rc})." >&2
  fi
  return "$rc"
}

run_sparse_dpo() {
  check_budget
  echo "=== Sparse DPO training for each mask ==="

  local mask_dir="${MASK_OUT_BASE}/${RUN_ID}"

  shopt -s nullglob
  local mask
  for mask in "${mask_dir}"/*.pt; do
    check_budget
    run_sparse_dpo_one_mask "$mask" || true
  done
  shopt -u nullglob
}

# Submit one Slurm GPU job per mask .pt (parallel sparse runs), then submit eval stage when all finish.
# PIPELINE_SPARSE_EVAL_DEPENDENCY: afterok (default) or afterany — whether eval stage waits for all sparse successes only.
# SPARSE_SLURM_TIME: per-job wall time (default 24:00:00).
launch_parallel_sparse_jobs_and_eval() {
  local mask_dir="${MASK_OUT_BASE}/${RUN_ID}"
  local dep_kind="${PIPELINE_SPARSE_EVAL_DEPENDENCY:-afterok}"
  local wall="${SPARSE_SLURM_TIME:-24:00:00}"
  local jids=()
  local mask

  mkdir -p logs

  shopt -s nullglob
  for mask in "${mask_dir}"/*.pt; do
    local mn safe
    mn=$(basename "$mask" .pt)
    safe=$(echo "$mn" | tr -c '[:alnum:]_' '_' | cut -c1-48)
    echo "Submitting parallel sparse job for: ${mn}"
    local jid
    jid=$(sbatch --parsable \
      --partition=gpu \
      --nodes=1 \
      --gres=gpu:h200:1 \
      --time="${wall}" \
      --mem=128G \
      --job-name="sp_${safe}" \
      --output="logs/sparse_${RUN_ID}_${safe}_%j.out" \
      --error="logs/sparse_${RUN_ID}_${safe}_%j.err" \
      --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}",PIPELINE_MASK_FILE="${mask}" \
      "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh")
    jids+=("$jid")
    echo "  → Slurm job ${jid}"
  done
  shopt -u nullglob

  local dep=""
  if [ "${#jids[@]}" -eq 0 ]; then
    echo "WARNING: No mask .pt files under ${mask_dir}; submitting eval stage after this launcher only."
    dep="${dep_kind}:${SLURM_JOB_ID}"
  else
    dep="${dep_kind}:${jids[0]}"
    local i
    for ((i = 1; i < ${#jids[@]}; i++)); do
      dep="${dep}:${jids[i]}"
    done
  fi

  echo "Submitting eval stage with dependency ${dep}"
  local ej
  ej=$(sbatch --parsable \
    --dependency="${dep}" \
    --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
    "${REPO_ROOT}/scripts/pipeline_stage_05_evals.sh")
  echo "Eval stage job id: ${ej}"
}

########################################
# 5. Submit eval jobs (base / dense / sparse)
########################################

submit_evals() {
  check_budget
  echo "=== Submitting eval jobs (base, dense, sparse) ==="

  # Helper: full benchmark suite unless EVAL_LIMIT is set (vLLM when EVAL_FORCE_HF_BACKEND=0 and available).
  submit_eval_job() {
    local model_path="$1"
    local out_dir="$2"
    local label="$3"
    echo "Submitting eval for ${label}: ${model_path}"
    local sbatch_args=(
      --export=ALL,FORCE_HF_BACKEND="${EVAL_FORCE_HF_BACKEND:-0}"
      scripts/run_evals_slurm.sh
      --model_path "$model_path"
      --trust_remote_code
      --output_dir "$out_dir"
    )
    if [ -n "${EVAL_LIMIT:-}" ]; then
      sbatch_args+=(--limit "$EVAL_LIMIT")
    fi
    sbatch "${sbatch_args[@]}" | awk '{print $4}'
  }

  # Baseline: raw HuggingFace Llama 3.1 8B Instruct (reference checkpoint for the suite)
  sbatch_id_baseline=$(submit_eval_job "$MODEL" "${EVAL_OUT_BASE}/${RUN_ID}/baseline_llama31_8b_it" "baseline Llama 3.1 8B Instruct (HF)")

  # Dense model eval: DPO_train saves to
  #   ${TRAIN_OUT_BASE}/${RUN_ID}/checkpoints/<model_dataset_dpo_dense>/checkpoint-<step>
  # We select the latest checkpoint directory.
  dense_model_path=""
  latest_ckpt_rel=$(ls -1dt "${TRAIN_OUT_BASE}/${RUN_ID}"/checkpoints/*/checkpoint-* 2>/dev/null | head -n 1 || true)
  if [ -n "${latest_ckpt_rel}" ] && [ -d "${latest_ckpt_rel}" ]; then
    dense_model_path="${latest_ckpt_rel}"
  fi

  if [ -n "${dense_model_path}" ]; then
    sbatch_id_dense=$(submit_eval_job "$dense_model_path" "${EVAL_OUT_BASE}/${RUN_ID}/dense" "dense model")
  else
    echo "WARNING: Dense checkpoint not found under ${TRAIN_OUT_BASE}/${RUN_ID}/checkpoints; skipping dense eval."
  fi

  # Sparse models eval:
  # sparse_dpo_efficiency uses:
  #   run_dir = <output_base_dir>/<run_name>
  #   final model at <run_dir>/final_model
  # and this pipeline sets output_base_dir to:
  #   ${SPARSE_OUT_BASE}/${RUN_ID}/${mask_name_without_ext}
  # so final model lives at:
  #   ${SPARSE_OUT_BASE}/${RUN_ID}/${mask_name_without_ext}/<run_name>/final_model
  local sparse_dir
  sparse_found=0
  for sparse_dir in "${SPARSE_OUT_BASE}/${RUN_ID}"/*/*/final_model; do
    if [ -d "${sparse_dir}" ]; then
      sparse_found=1
      local tag
      tag=$(basename "$(dirname "$sparse_dir")")
      sbatch_id_sparse=$(submit_eval_job "${sparse_dir}" "${EVAL_OUT_BASE}/${RUN_ID}/sparse_${tag}" "sparse model ${tag}")
    fi
  done
  if [ "${sparse_found}" -eq 0 ]; then
    echo "WARNING: No sparse final_model directories found under ${SPARSE_OUT_BASE}/${RUN_ID}; skipping sparse evals."
  fi

  echo "Eval jobs submitted (if any). Check with: squeue -u \$USER"
}


#!/bin/bash
# Full single-GPU pipeline:
# 1) Dense DPO training
# 2) Mask building (warm + cold)
# 3) Sparse DPO training for each mask
# 4) Submit eval jobs for base / dense / sparse models
#
# Usage (from repo root):
#   sbatch scripts/run_full_pipeline.sh
#
# Notes:
# - Designed for a single A100/H100 with <= 8h walltime.
# - Uses rl_casino (training) and rl_casino_eval (eval) Conda envs.

#SBATCH --job-name=full_pipeline
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

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

RUN_ID="${RUN_ID_OVERRIDE:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
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

# Model / dataset configuration
MODEL="google/gemma-3-270m-it"
DPO_DATASETS=("light-r1")     # dataset keys used by dataset_registry
NUM_STEPS_DPO=50              # keep small enough to fit in 8h budget
SUBSET_DPO=256

SPARSITY_LIST=("97.5")        # can extend later
TARGET_STEP_DPO=50            # must match a checkpoint step from DPO

EVAL_LIMIT=100                # number of examples per benchmark for sanity checks

# Timeouts (seconds)
TRAIN_TIMEOUT_PER_DATASET=$((3 * 60 * 60))   # 3h per dense DPO run
MASK_TIMEOUT=$((2 * 60 * 60))                # 2h for all masks
SPARSE_TIMEOUT_PER_MASK=$((2 * 60 * 60))     # 2h per sparse run

GLOBAL_MAX_SECONDS=$((8 * 60 * 60))          # 8h safety budget
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

########################################
# 1. Env setup
########################################

echo "Using training env: ${TRAIN_ENV}"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found at ${TRAIN_PY}" >&2
  exit 1
fi
# Prefer explicit env binaries (works when conda.sh lives somewhere nonstandard on HPC)
export PATH="${TRAIN_ENV}/bin:${PATH}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "Environment check (training):"
"$TRAIN_PY" -c "import torch, trl; print(f'Torch: {torch.__version__}, TRL: {trl.__version__}')" || {
  echo "WARNING: Could not import torch+trl in training env; ensure requirements are installed." >&2
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

    timeout --signal=TERM --kill-after=60 "${TRAIN_TIMEOUT_PER_DATASET}" \
      "$TRAIN_PY" src/full_training/DPO_train.py \
        --model_name "$MODEL" \
        --dataset "$ds" \
        --num_steps "$NUM_STEPS_DPO" \
        --subset_size "$SUBSET_DPO" \
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

  # Reconstruct delta log dir pattern to match DPO_train path logic:
  # DELTA_LOG_DIR = os.path.join(BASE_DIR, "deltas", f"{MODEL_NAME_SANITIZED}_{DATASET_SANITIZED}")
  # Dataset sanitization is done in dataset_registry; we approximate by replacing '-' with '_'.

  local model_sanitized ds_sanitized delta_root delta_dir
  model_sanitized=$(echo "$MODEL" | tr '/-' '_' | tr '[:upper:]' '[:lower:]')
  ds_sanitized=$(echo "$ds" | tr '-' '_')
  delta_root="${TRAIN_OUT_BASE}/${RUN_ID}/deltas"
  delta_dir="${delta_root}/${model_sanitized}_${ds_sanitized}"

  echo "Using DPO delta dir: ${delta_dir}"

  mkdir -p "${MASK_OUT_BASE}/${RUN_ID}"

  (
    set -e
    cd "$REPO_ROOT"

    timeout --signal=TERM --kill-after=60 "${MASK_TIMEOUT}" bash -c '
      set -e
      MASK_BASE="'"${MASK_OUT_BASE}/${RUN_ID}"'"
      DELTA_DIR="'"${delta_dir}"'"

      echo "Warm-start masks from ${DELTA_DIR}"
      for method in magnitude momentum fisher; do
        for sparsity in '"${SPARSITY_LIST[@]}"'; do
          echo "-> Warm mask: ${method}, sparsity=${sparsity}"
          '"$TRAIN_PY"' src/warm_start/even_better_mask_finder.py \
            --delta_log_dir "$DELTA_DIR" \
            --method "$method" \
            --sparsity_percent "$sparsity" \
            --target_step '"${TARGET_STEP_DPO}"' \
            --mlp_only \
            --output_file "${MASK_BASE}/warm_${method}_${model_sanitized}_${ds_sanitized}_sparsity${sparsity}pct_step'"${TARGET_STEP_DPO}"'.pt"
        done
      done

      echo "Cold-start Fisher mask"
      for sparsity in '"${SPARSITY_LIST[@]}"'; do
        '"$TRAIN_PY"' src/cold_start/cold_mask_finder.py \
          --model_name "'"${MODEL}"'" \
          --dataset_name "qihoo360/Light-R1-DPOData" \
          --sparsity_percent "$sparsity" \
          --n_calibration_samples 256 \
          --mlp_only \
          --output_file "${MASK_BASE}/cold_fisher_${model_sanitized}_sparsity${sparsity}pct_n256.pt"
      done

      echo "Cold-start CAV mask"
      for sparsity in '"${SPARSITY_LIST[@]}"'; do
        '"$TRAIN_PY"' src/cold_start/cav_cold_mask_finder.py \
          --model_name "'"${MODEL}"'" \
          --dataset_name "qihoo360/Light-R1-DPOData" \
          --method cav \
          --sparsity_percent "$sparsity" \
          --subset_size 256 \
          --num_batches 16 \
          --mlp_only \
          --output_file "${MASK_BASE}/cold_cav_${model_sanitized}_sparsity${sparsity}pct.pt"
      done
    ' 2>&1 | tee "logs/full_pipeline_masks_${RUN_ID}.log"
  )

  local rc=$?
  if (( rc != 0 )); then
    echo "ERROR: Mask building failed (exit ${rc}). Aborting pipeline." >&2
    exit 1
  fi
}

########################################
# 4. Sparse DPO training
########################################

run_sparse_dpo() {
  check_budget
  echo "=== Sparse DPO training for each mask ==="

  local ds="${DPO_DATASETS[0]}"
  local mask_dir="${MASK_OUT_BASE}/${RUN_ID}"
  local model_sanitized ds_sanitized

  model_sanitized=$(echo "$MODEL" | tr '/-' '_' | tr '[:upper:]' '[:lower:]')
  ds_sanitized=$(echo "$ds" | tr '-' '_')

  shopt -s nullglob
  local mask
  for mask in "${mask_dir}"/*.pt; do
    check_budget
    local mask_name
    mask_name=$(basename "$mask")
    echo "-> Sparse DPO run for mask=${mask_name}"

    local out_base cache_dir run_name
    out_base="${SPARSE_OUT_BASE}/${RUN_ID}/${mask_name%.pt}"
    cache_dir="/scratch/biggs.s/hf_cache/datasets"
    run_name="fullpipe_sparse_${mask_name%.*}_${RUN_ID}"

    mkdir -p "$out_base"

    timeout --signal=TERM --kill-after=60 "${SPARSE_TIMEOUT_PER_MASK}" \
      "$TRAIN_PY" src/full_training/sparse_dpo_efficiency.py \
        --model_name "$MODEL" \
        --checkpoint None \
        --mask "$mask" \
        --n_steps "$NUM_STEPS_DPO" \
        --batch_size 4 \
        --grad_accum 4 \
        --subset_size "$SUBSET_DPO" \
        --optimizer sparse_adamw \
        --block_size 32 \
        --mlp_only \
        --use_wandb \
        --save_csv \
        --output_base_dir "$out_base" \
        --dataset_cache_dir "$cache_dir" \
        --run_name "$run_name" 2>&1 | tee "logs/full_pipeline_sparse_${mask_name%.*}_${RUN_ID}.log"

    local rc=$?
    if (( rc != 0 )); then
      echo "WARNING: Sparse DPO failed for mask ${mask_name} (exit ${rc}). Continuing to next mask." >&2
    fi
  done
  shopt -u nullglob
}

########################################
# 5. Submit eval jobs (base / dense / sparse)
########################################

submit_evals() {
  check_budget
  echo "=== Submitting eval jobs (base, dense, sparse) ==="

  # Helper: submit eval with conservative backend/options for compatibility.
  submit_eval_job() {
    local model_path="$1"
    local out_dir="$2"
    local label="$3"
    echo "Submitting eval for ${label}: ${model_path}"
    sbatch --export=ALL,FORCE_HF_BACKEND=1 scripts/run_evals_slurm.sh \
      --model_path "$model_path" \
      --limit "$EVAL_LIMIT" \
      --trust_remote_code \
      --output_dir "$out_dir" | awk '{print $4}'
  }

  # Base model eval
  sbatch_id_base=$(submit_eval_job "$MODEL" "${EVAL_OUT_BASE}/${RUN_ID}/base" "base model")

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

########################################
# 6. Orchestration
########################################

echo "===== FULL PIPELINE START (${RUN_ID}) ====="
run_dense_dpo
run_masks
run_sparse_dpo
submit_evals
echo "===== FULL PIPELINE COMPLETE (${RUN_ID}) ====="


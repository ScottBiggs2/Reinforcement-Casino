#!/usr/bin/env bash
# Re-queue the 12 downstream training jobs from an *existing* mask run (no mask generation, no GPU).
# Use when masks already exist under MASK_DIR but training submissions failed, were cancelled, or you
# need to drop the afterok dependency on the orchestrator job.
#
# From repo root on a login node (Explorer):
#   export MASK_RUN_ID=orch_masks_6262915   # required; must match the directory under rl_casino_masks
#   export ORCH_USE_TRAIN_AUTO_RESUME=1
#   # match whatever you used for the full orchestrator (paths, steps, ORCH_TRAIN_* , etc.)
#   bash scripts/orchestrate_queue_training_only.sh
#
# Optional: ORCH_SBATCH_DEPENDENCY=afterok:12345   # only if you want an explicit dependency
# If the scheduler rejects bursts of sbatch: ORCH_SLEEP_BETWEEN_SUBMIT_SEC=3
# Paths must match the mask job: MASK_OUT_BASE, SCRATCH_USER_ROOT, MODEL, SPARSITY_PERCENT, GRPO_DATASET_HF.
# Override any single file: ORCH_MASK_RANDOM_FILE=/abs/path/random_....pt
#
# Keep defaults in sync with scripts/orchestrate_masks_then_queue_dpo_grpo.slurm.
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

if [ -z "${MASK_RUN_ID:-}" ]; then
  echo "ERROR: export MASK_RUN_ID=orch_masks_<jobid> (directory name under rl_casino_masks)." >&2
  exit 1
fi

########################################
# Config — mirror orchestrate_masks_then_queue_dpo_grpo.slurm
########################################
SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"
export TRAIN_OUT_BASE="${TRAIN_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_train}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"
export GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"

export HF_TOKEN="${HF_TOKEN:-}"
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

export DPO_DS_KEY="${DPO_DS_KEY:-tulu3}"
export COLD_DATASET_HF="${COLD_DATASET_HF:-allenai/llama-3.1-tulu-3-8b-preference-mixture}"
export DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"
export COLD_DATASET_HF_LIGHT_R1="${COLD_DATASET_HF_LIGHT_R1:-qihoo360/Light-R1-DPOData}"
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export DPO_SNIP_OBJECTIVE="${DPO_SNIP_OBJECTIVE:-dpo_preference}"
export GRPO_SNIP_OBJECTIVE="${GRPO_SNIP_OBJECTIVE:-lm}"
export DPO_BETA="${DPO_BETA:-0.1}"
export GRPO_SNIP_PREFERENCE_BETA="${GRPO_SNIP_PREFERENCE_BETA:-1.0}"

export MASK_DIR="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}/${MASK_RUN_ID}"
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"

export DPO_DENSE_RUN_ID="${DPO_DENSE_RUN_ID:-dpo5k_dense_${DPO_DS_KEY}}"
export DPO_SPARSE_RANDOM_RUN_ID="${DPO_SPARSE_RANDOM_RUN_ID:-dpo5k_sparse_random_${DPO_DS_KEY}}"
export DPO_SPARSE_CAV_RUN_ID="${DPO_SPARSE_CAV_RUN_ID:-dpo5k_sparse_cav_${DPO_DS_KEY}}"
export DPO_SPARSE_SNIP_RUN_ID="${DPO_SPARSE_SNIP_RUN_ID:-dpo5k_sparse_snip_${DPO_DS_KEY}}"
export DPO_DENSE_RUN_ID_LIGHT_R1="${DPO_DENSE_RUN_ID_LIGHT_R1:-dpo5k_dense_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1="${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1:-dpo5k_sparse_random_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_CAV_RUN_ID_LIGHT_R1="${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1:-dpo5k_sparse_cav_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1="${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1:-dpo5k_sparse_snip_${DPO_DS_KEY_LIGHT_R1}}"

export GRPO_DENSE_RUN_SLUG="${GRPO_DENSE_RUN_SLUG:-llama31_${GRPO_DATASET_HF##*/}_grpo_dense_v1}"
export GRPO_SPARSE_RANDOM_RUN_NAME="${GRPO_SPARSE_RANDOM_RUN_NAME:-grpo_sparse_random_sp975_v1}"
export GRPO_SPARSE_CAV_RUN_NAME="${GRPO_SPARSE_CAV_RUN_NAME:-grpo_sparse_cav_sp975_v1}"
export GRPO_SPARSE_SNIP_RUN_NAME="${GRPO_SPARSE_SNIP_RUN_NAME:-grpo_sparse_snip_${GRPO_SNIP_OBJECTIVE}_sp975_v1}"

export NUM_STEPS_DPO="${NUM_STEPS_DPO:-5000}"
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-50}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"

export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-5000}"
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-2048}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"

export ORCH_USE_TRAIN_AUTO_RESUME="${ORCH_USE_TRAIN_AUTO_RESUME:-0}"
export MAX_AUTO_RESUME="${MAX_AUTO_RESUME:-8}"
export ORCH_TRAIN_PARTITION="${ORCH_TRAIN_PARTITION:-gpu}"
export ORCH_TRAIN_GRES="${ORCH_TRAIN_GRES:-gpu:h200:1}"
export ORCH_TRAIN_MEM="${ORCH_TRAIN_MEM:-128G}"
export ORCH_TRAIN_CPUS="${ORCH_TRAIN_CPUS:-8}"
export ORCH_TRAIN_TIME_DPO="${ORCH_TRAIN_TIME_DPO:-07:45:00}"
export ORCH_TRAIN_TIME_GRPO="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"
if [ "${ORCH_USE_TRAIN_AUTO_RESUME}" = "1" ] && [ -n "${ORCH_TRAIN_SOFT_SECONDS:-}" ]; then
  export AUTO_RESUME_SOFT_SECONDS="${ORCH_TRAIN_SOFT_SECONDS}"
fi

unset WANDB_DISABLED 2>/dev/null || true
unset WANDB_SILENT 2>/dev/null || true
export WANDB_MODE="online"

sanitize_model_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s//-/_}"
  s="${s,,}"
  s="$(echo "$s" | tr -c '[:alnum:]_' '_' )"
  while [[ "$s" == *"__"* ]]; do s="${s//__/_}"; done
  s="${s##_}"
  s="${s%%_}"
  echo "$s"
}

MODEL_SANITIZED="$(sanitize_model_name "$MODEL")"
export RANDOM_MASK_SEED="${RANDOM_MASK_SEED:-42}"

# Exact names match orchestrate_masks_then_queue_dpo_grpo.slurm (random uses MODEL with /→_ only; not full sanitize).
export MASK_RANDOM="${MASK_DIR}/random_${MODEL//\//_}_sparsity${SPARSITY_PERCENT}pct_seed${RANDOM_MASK_SEED}.pt"
export MASK_CAV="${MASK_DIR}/cold_cav_${MODEL_SANITIZED}_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_SNIP="${MASK_DIR}/cold_snip_${MODEL_SANITIZED}_${DPO_DS_KEY}_sparsity${SPARSITY_PERCENT}pct_${DPO_SNIP_OBJECTIVE}.pt"
export MASK_CAV_LIGHT_R1="${MASK_DIR}/cold_cav_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_SNIP_LIGHT_R1="${MASK_DIR}/cold_snip_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct_${DPO_SNIP_OBJECTIVE}.pt"
export MASK_GRPO_RANDOM="${MASK_RANDOM}"
export MASK_GRPO_CAV="${GRPO_MASK_DIR}/grpo_cav_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}.pt"
export MASK_GRPO_SNIP="${GRPO_MASK_DIR}/grpo_snip_${GRPO_SNIP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}.pt"

# Optional: absolute paths if defaults do not match (e.g. different MODEL/SPARSITY in shell than mask run).
if [ -n "${ORCH_MASK_RANDOM_FILE:-}" ]; then export MASK_RANDOM="${ORCH_MASK_RANDOM_FILE}"; MASK_GRPO_RANDOM="${MASK_RANDOM}"; export MASK_GRPO_RANDOM; fi
if [ -n "${ORCH_MASK_CAV_FILE:-}" ]; then export MASK_CAV="${ORCH_MASK_CAV_FILE}"; fi
if [ -n "${ORCH_MASK_SNIP_FILE:-}" ]; then export MASK_SNIP="${ORCH_MASK_SNIP_FILE}"; fi
if [ -n "${ORCH_MASK_CAV_LIGHT_R1_FILE:-}" ]; then export MASK_CAV_LIGHT_R1="${ORCH_MASK_CAV_LIGHT_R1_FILE}"; fi
if [ -n "${ORCH_MASK_SNIP_LIGHT_R1_FILE:-}" ]; then export MASK_SNIP_LIGHT_R1="${ORCH_MASK_SNIP_LIGHT_R1_FILE}"; fi
if [ -n "${ORCH_MASK_GRPO_CAV_FILE:-}" ]; then export MASK_GRPO_CAV="${ORCH_MASK_GRPO_CAV_FILE}"; fi
if [ -n "${ORCH_MASK_GRPO_SNIP_FILE:-}" ]; then export MASK_GRPO_SNIP="${ORCH_MASK_GRPO_SNIP_FILE}"; fi

# If exact random path missing, pick the lone random_*_seed<RANDOM_MASK_SEED>.pt in MASK_DIR (handles MODEL/SPARSITY env drift).
if [ ! -f "$MASK_RANDOM" ]; then
  _pick=""
  shopt -s nullglob
  _cand=( "${MASK_DIR}"/random_*_seed"${RANDOM_MASK_SEED}".pt )
  shopt -u nullglob
  if [ "${#_cand[@]}" -eq 1 ] && [ -f "${_cand[0]}" ]; then
    _pick="${_cand[0]}"
  fi
  if [ -n "$_pick" ]; then
    echo "NOTE: using discovered random mask (expected path missing): ${_pick}" >&2
    MASK_RANDOM="${_pick}"
    MASK_GRPO_RANDOM="${MASK_RANDOM}"
    export MASK_RANDOM MASK_GRPO_RANDOM
  fi
fi

orch_mask_precheck_fail() {
  local missing="$1"
  echo "ERROR: expected mask file missing: ${missing}" >&2
  echo "  MASK_DIR=${MASK_DIR}  MODEL=${MODEL}  SPARSITY_PERCENT=${SPARSITY_PERCENT}  DPO_DS_KEY=${DPO_DS_KEY}" >&2
  if [ ! -d "${MASK_DIR}" ]; then
    echo "  MASK_DIR is not a directory. Set MASK_OUT_BASE and/or SCRATCH_USER_ROOT to match the orchestrator run." >&2
  else
    echo "  Files in MASK_DIR (random + cold_* only):" >&2
    ls -1 "${MASK_DIR}" 2>/dev/null | grep -E '^(random_|cold_)' || ls -1 "${MASK_DIR}" >&2
  fi
  echo "  Fix: export MODEL, SPARSITY_PERCENT, DPO_DS_KEY, MASK_OUT_BASE to match the mask job, or set ORCH_MASK_*_FILE overrides (see script header)." >&2
  exit 1
}

if [ ! -d "${MASK_DIR}" ]; then
  orch_mask_precheck_fail "${MASK_DIR}/"
fi

for f in "$MASK_RANDOM" "$MASK_CAV" "$MASK_SNIP" "$MASK_CAV_LIGHT_R1" "$MASK_SNIP_LIGHT_R1" "$MASK_GRPO_CAV" "$MASK_GRPO_SNIP"; do
  if [ ! -f "$f" ]; then
    orch_mask_precheck_fail "$f"
  fi
done

echo "=== Queue training only (masks OK under MASK_DIR=${MASK_DIR}) ==="
echo "ORCH_SBATCH_DEPENDENCY=${ORCH_SBATCH_DEPENDENCY:-<none>}"

ORCH_TRAIN_ENTRY="${REPO_ROOT}/scripts/orchestrate_training_child_entry.sh"

_orch_train_bases_export() {
  echo "NUM_STEPS_DPO=${NUM_STEPS_DPO},GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS},TRAIN_OUT_BASE=${TRAIN_OUT_BASE},SPARSE_OUT_BASE=${SPARSE_OUT_BASE},GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE},GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE},GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS},GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
}

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

submit_training_job() {
  local label="$1"
  local inner_script="$2"
  local auto_mode="$3"
  local export_extra="$4"
  local job_tag="${5:-$auto_mode}"

  local bases_steps jid
  bases_steps="$(_orch_train_bases_export)"
  local -a dep_args=()
  if [ -n "${ORCH_SBATCH_DEPENDENCY:-}" ]; then
    dep_args=( --dependency="${ORCH_SBATCH_DEPENDENCY}" )
  fi

  if [ "${ORCH_USE_TRAIN_AUTO_RESUME:-0}" != "1" ]; then
    jid=$(sbatch --parsable "${dep_args[@]}" \
      --export=ALL,"${export_extra},${bases_steps}" \
      "${inner_script}")
    echo "${label}: ${jid}"
    if [ -n "${ORCH_SLEEP_BETWEEN_SUBMIT_SEC:-}" ]; then sleep "${ORCH_SLEEP_BETWEEN_SUBMIT_SEC}"; fi
    return
  fi

  local part gres mem time_wall out_log err_log jname full_export
  local -a cpus_args=()
  part="${ORCH_TRAIN_PARTITION:-gpu}"
  gres="${ORCH_TRAIN_GRES:-gpu:h200:1}"
  mem="${ORCH_TRAIN_MEM:-128G}"

  case "${auto_mode}" in
    dense_dpo|sparse_dpo)
      time_wall="${ORCH_TRAIN_TIME_DPO:-07:45:00}"
      ;;
    dense_grpo|sparse_grpo)
      time_wall="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"
      cpus_args=( --cpus-per-task="${ORCH_TRAIN_CPUS:-8}" )
      ;;
    *)
      echo "ERROR: unknown auto-resume mode tag: ${auto_mode}" >&2
      exit 1
      ;;
  esac

  case "${auto_mode}" in
    dense_dpo)
      out_log='logs/pipeline_%j_p1_dense.out'
      err_log='logs/pipeline_%j_p1_dense.err'
      ;;
    sparse_dpo)
      out_log='logs/pipeline_sparse_one_mask_%j.out'
      err_log='logs/pipeline_sparse_one_mask_%j.err'
      ;;
    dense_grpo|sparse_grpo)
      out_log='logs/grpo_openr1_%j.out'
      err_log='logs/grpo_openr1_%j.err'
      ;;
  esac

  full_export="ALL,${export_extra},${bases_steps},ORCH_TRAIN_INNER=${inner_script},AUTO_RESUME_MODE=${auto_mode},ORCH_USE_TRAIN_AUTO_RESUME=1"
  if [ -n "${AUTO_RESUME_SOFT_SECONDS:-}" ]; then
    full_export+=",AUTO_RESUME_SOFT_SECONDS=${AUTO_RESUME_SOFT_SECONDS}"
  fi
  full_export+=",MAX_AUTO_RESUME=${MAX_AUTO_RESUME}"

  jname="$(orch_sanitize_job_name "orch_${job_tag}")"

  jid=$(sbatch --parsable "${dep_args[@]}" \
    --partition="${part}" --nodes=1 --ntasks=1 \
    --gres="${gres}" --mem="${mem}" --time="${time_wall}" \
    "${cpus_args[@]}" \
    --job-name="${jname}" \
    --output="${out_log}" --error="${err_log}" \
    --export="${full_export}" \
    "${ORCH_TRAIN_ENTRY}")
  echo "${label}: ${jid}"
  if [ -n "${ORCH_SLEEP_BETWEEN_SUBMIT_SEC:-}" ]; then sleep "${ORCH_SLEEP_BETWEEN_SUBMIT_SEC}"; fi
}

# echo ""
# echo "--- DPO dense (Tulu3) ---"
# submit_training_job "DPO dense (stage 1 only) RUN_ID=${DPO_DENSE_RUN_ID}" \
#   "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh" \
#   dense_dpo \
#   "PIPELINE_CHAIN_NEXT_STAGE=0,PIPELINE_RUN_ID=${DPO_DENSE_RUN_ID},RUN_ID=${DPO_DENSE_RUN_ID},DPO_DATASET_KEY=tulu3" \
#   "dpo_dense_${DPO_DENSE_RUN_ID}"

# echo ""
# echo "--- DPO sparse Tulu3 ---"
# submit_training_job "DPO sparse random RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID}" \
#   "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#   sparse_dpo \
#   "PIPELINE_RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID},RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID},PIPELINE_MASK_FILE=${MASK_RANDOM},DPO_DATASET_KEY=tulu3" \
#   "dpo_sp_rand_${DPO_SPARSE_RANDOM_RUN_ID}"

# submit_training_job "DPO sparse CAV RUN_ID=${DPO_SPARSE_CAV_RUN_ID}" \
#   "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#   sparse_dpo \
#   "PIPELINE_RUN_ID=${DPO_SPARSE_CAV_RUN_ID},RUN_ID=${DPO_SPARSE_CAV_RUN_ID},PIPELINE_MASK_FILE=${MASK_CAV},DPO_DATASET_KEY=tulu3" \
#   "dpo_sp_cav_${DPO_SPARSE_CAV_RUN_ID}"

# submit_training_job "DPO sparse SNIP RUN_ID=${DPO_SPARSE_SNIP_RUN_ID}" \
#   "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#   sparse_dpo \
#   "PIPELINE_RUN_ID=${DPO_SPARSE_SNIP_RUN_ID},RUN_ID=${DPO_SPARSE_SNIP_RUN_ID},PIPELINE_MASK_FILE=${MASK_SNIP},DPO_DATASET_KEY=tulu3" \
#   "dpo_sp_snip_${DPO_SPARSE_SNIP_RUN_ID}"

# echo ""
# echo "--- DPO dense + sparse (Light-R1) ---"
# submit_training_job "DPO dense Light-R1 RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1}" \
#   "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh" \
#   dense_dpo \
#   "PIPELINE_CHAIN_NEXT_STAGE=0,PIPELINE_RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1},RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#   "dpo_dense_lr1_${DPO_DENSE_RUN_ID_LIGHT_R1}"

# submit_training_job "DPO sparse random Light-R1 RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1}" \
#   "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#   sparse_dpo \
#   "PIPELINE_RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_RANDOM},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#   "dpo_sp_rand_lr1_${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1}"

# submit_training_job "DPO sparse CAV Light-R1 RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}" \
#   "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#   sparse_dpo \
#   "PIPELINE_RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_CAV_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#   "dpo_sp_cav_lr1_${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}"

# submit_training_job "DPO sparse SNIP Light-R1 RUN_ID=${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1}" \
#   "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#   sparse_dpo \
#   "PIPELINE_RUN_ID=${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_SNIP_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#   "dpo_sp_snip_lr1_${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1}"

echo ""
echo "--- GRPO dense ---"
submit_training_job "GRPO dense slug=${GRPO_DENSE_RUN_SLUG}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
  dense_grpo \
  "GRPO_MODE=dense,GRPO_RUN_SLUG=${GRPO_DENSE_RUN_SLUG},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
  "grpo_dense_${GRPO_DENSE_RUN_SLUG}"

echo ""
echo "--- GRPO sparse ---"
submit_training_job "GRPO sparse random name=${GRPO_SPARSE_RANDOM_RUN_NAME}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
  sparse_grpo \
  "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_RANDOM},GRPO_RUN_NAME=${GRPO_SPARSE_RANDOM_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
  "grpo_sp_rand_${GRPO_SPARSE_RANDOM_RUN_NAME}"

submit_training_job "GRPO sparse CAV name=${GRPO_SPARSE_CAV_RUN_NAME}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
  sparse_grpo \
  "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_CAV},GRPO_RUN_NAME=${GRPO_SPARSE_CAV_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
  "grpo_sp_cav_${GRPO_SPARSE_CAV_RUN_NAME}"

submit_training_job "GRPO sparse SNIP name=${GRPO_SPARSE_SNIP_RUN_NAME}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
  sparse_grpo \
  "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_SNIP},GRPO_RUN_NAME=${GRPO_SPARSE_SNIP_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
  "grpo_sp_snip_${GRPO_SPARSE_SNIP_RUN_NAME}"

echo ""
echo "=== Done. Monitor: squeue -u \$USER ==="

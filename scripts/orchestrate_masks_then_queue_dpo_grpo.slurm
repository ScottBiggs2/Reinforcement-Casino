#!/usr/bin/env bash
# One short GPU job that:
#   1) Generates masks on GPU: shared random baseline + Light-R1 Wanda+OWL + Light-R1 SNIP/GRaSP +
#      GRPO Wanda+OWL + GRPO SNIP/GRaSP (plus optional ORCH_SCORE_SNR / ORCH_MASK_GRANULARITY flags).
#   2) As each mask file is written, submits the matching downstream training job via sbatch (no
#      Slurm afterok on this orchestrator job — children start independently once their mask exists).
#
# Light-R1 DPO: random sparse, dense, Wanda+OWL sparse, SNIP sparse, GRaSP sparse.
# GRPO: random sparse, Wanda+OWL sparse, SNIP sparse, GRaSP sparse.
# Export MASK_RUN_ID / paths before sbatch if you only need to re-queue training:
#   bash scripts/orchestrate_queue_training_only.sh
#
# Submit from repo root:
#   cd /path/to/rl_casino && sbatch scripts/orchestrate_masks_then_queue_dpo_grpo.slurm
#
# If masks already exist and you only need to (re)submit the 12 training jobs (no GPU, no afterok):
#   export MASK_RUN_ID=orch_masks_<jobid>
#   bash scripts/orchestrate_queue_training_only.sh
#
# Override any knobs by exporting env vars before sbatch (see "Config" section below).
# Orchestrator-only: ORCH_USE_TRAIN_AUTO_RESUME, ORCH_TRAIN_SOFT_SECONDS, ORCH_TRAIN_* resource overrides.
#
# Long training jobs: optional automatic resume (soft timeout + chained sbatch) is documented in
# scripts/README.md ("Automatic wall-time resume"). For this orchestrator, export ORCH_USE_TRAIN_AUTO_RESUME=1
# (and optionally ORCH_TRAIN_SOFT_SECONDS, ORCH_TRAIN_PARTITION, …) so queued children use
# scripts/orchestrate_training_child_entry.sh → train_with_auto_resume.sh with Slurm resources on the CLI.
# Or submit pipeline_stage_01_dense.sh, pipeline_sparse_one_mask.sh, or grpo_openr1_llama31_slurm.sh
# directly with USE_TRAIN_WITH_AUTO_RESUME=1 as in the README.
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# (Host RAM for the Python process — not GPU VRAM. Mask OOM is typically CUDA OOM, not this limit.)
#SBATCH --time=04:00:00
#SBATCH --job-name=orch_masks_queue
#SBATCH --output=logs/orch_masks_queue_%j.out
#SBATCH --error=logs/orch_masks_queue_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

########################################
# Config (override before sbatch)
# Optional: source scripts/export_pipeline_step_targets.sh to pin NUM_STEPS_DPO=500 and
# GRPO_TARGET_STEPS=ORCH_GRPO_TRAIN_STEPS=1000 in your shell before sbatch.
########################################

# Scratch / env
SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"

# Training output roots (pipeline_common.sh / grpo_openr1_llama31_slurm.sh). Export before queueing
# children so training_resume_probe in train_with_auto_resume.sh can resolve checkpoint paths.
export TRAIN_OUT_BASE="${TRAIN_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_train}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"
export GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"

# Model + datasets
export HF_TOKEN="${HF_TOKEN:-}" # required for gated models (e.g. Llama). If unset, HF may still work if token is cached.
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# DPO mask generation config (tulu3 defaults)
export DPO_DS_KEY="${DPO_DS_KEY:-tulu3}"  # dataset registry key for DPO mode
export COLD_DATASET_HF="${COLD_DATASET_HF:-allenai/llama-3.1-tulu-3-8b-preference-mixture}"

# Light-R1 DPO (same model + hyperparameters; separate CAV/SNIP calibration — random mask shared)
export DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"
export COLD_DATASET_HF_LIGHT_R1="${COLD_DATASET_HF_LIGHT_R1:-qihoo360/Light-R1-DPOData}"

# GRPO mask generation config (OpenR1 math defaults)
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

# Mask knobs
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# CAV implementation + safety knobs
# - cold_v2_all_params: unified CAV v2 (all params suspect; global selector; coverage gated)
# - legacy: use existing scripts (DPO via cav_cold_mask_finder, GRPO via inference_mask_finder)
export ORCH_CAV_IMPL="${ORCH_CAV_IMPL:-cold_v2_all_params}"
# Opt-in rank-collapse protection (applies to broadcast-based scoring via * |W|).
export ORCH_CAV_WEIGHT_ABS="${ORCH_CAV_WEIGHT_ABS:-0}"
# Also score embedding matrices.
# We default this ON for CAV v2 so the coverage gate includes `model.embed_tokens.weight`
# (aligns with warm-start "all params suspect" masks).
export ORCH_CAV_INCLUDE_EMBEDS="${ORCH_CAV_INCLUDE_EMBEDS:-1}"

# inference_mask_finder batch size (all mask methods that pass --batch_size). Lower if SNIP/CAV OOMs.
# Sequence caps for masks follow DPO_MAX_LENGTH / GRPO_MAX_PROMPT_LENGTH (exported below for downstream;
# mask steps use the same vars in the commands in section "1) Mask generation").
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-4}"

# Calibration sizes
export COLD_CAV_SUBSET="${COLD_CAV_SUBSET:-1024}"
export COLD_CAV_NUM_BATCHES="${COLD_CAV_NUM_BATCHES:-32}"
export GRPO_N_SAMPLES="${GRPO_N_SAMPLES:-64}"

# SNIP/GRaSP objectives
export DPO_SNIP_OBJECTIVE="${DPO_SNIP_OBJECTIVE:-dpo_preference}"
export GRPO_SNIP_OBJECTIVE="${GRPO_SNIP_OBJECTIVE:-lm}"

export GRASP_OBJECTIVE="${GRASP_OBJECTIVE:-dpo_preference}"
export GRPO_GRASP_OBJECTIVE="${GRPO_GRASP_OBJECTIVE:-lm}"

# DPO preference beta for SNIP/GRaSP (objective=dpo_preference)
export DPO_BETA="${DPO_BETA:-0.1}"
export GRPO_BETA="${GRPO_BETA:-1.0}"
export GRPO_SNIP_PREFERENCE_BETA="${GRPO_SNIP_PREFERENCE_BETA:-$GRPO_BETA}"
export GRPO_GRASP_PREFERENCE_BETA="${GRPO_GRASP_PREFERENCE_BETA:-$GRPO_BETA}"

# Optional score reweighting (SNIP / GRaSP): default off.
# These are forwarded as CLI flags to inference_mask_finder.py for direct A/B comparison.
export ORCH_SCORE_SNR="${ORCH_SCORE_SNR:-off}" # off|per_tensor|per_weight
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}" # identity|log1p|clamp
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_CLAMP_MIN="${ORCH_SCORE_SNR_CLAMP_MIN:-0.0}"
export ORCH_SCORE_SNR_CLAMP_MAX="${ORCH_SCORE_SNR_CLAMP_MAX:-50.0}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-96.0}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}" # 1 to override RAM guard

# SNIP/GRaSP mask layout: element (default) vs block (pool to B×B blocks then expand; BSR-friendly).
export ORCH_MASK_GRANULARITY="${ORCH_MASK_GRANULARITY:-element}" # element|block
export ORCH_MASK_BLOCK_SIZE="${ORCH_MASK_BLOCK_SIZE:-256}"
export ORCH_MASK_BLOCK_REDUCTION="${ORCH_MASK_BLOCK_REDUCTION:-mean}" # mean|max
# SNIP/GRaSP output filename suffix so element vs block never clobber each other:
# - element (default): empty suffix (legacy filenames).
# - block: defaults to "_block${ORCH_MASK_BLOCK_SIZE}" (e.g. _block256).
# - Override anytime: export ORCH_MASK_FNAME_TAG=_elem   or   _block128_max
case "${ORCH_MASK_GRANULARITY:-element}" in
  block)
    ORCH_MASK_FNAME_TAG="${ORCH_MASK_FNAME_TAG:-_block${ORCH_MASK_BLOCK_SIZE}}"
    ;;
  *)
    ORCH_MASK_FNAME_TAG="${ORCH_MASK_FNAME_TAG:-}"
    ;;
esac

# Output locations for masks
export MASK_RUN_ID="${MASK_RUN_ID:-orch_masks_${SLURM_JOB_ID:-manual}}"
export MASK_DIR="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}/${MASK_RUN_ID}"
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"

########################################
# Oracle checkpoint directory for Light-R1
export ORACLE_FINAL_CKPT_DIR="${ORACLE_FINAL_CKPT_DIR:-/scratch/biggs.s/rl_casino_train/dpo5k_dense_light-r1/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500}"

# Oracle checkpoint directory for Tulu3
export ORACLE_FINAL_CKPT_DIR_TULU3="${ORACLE_FINAL_CKPT_DIR_TULU3:-/scratch/biggs.s/rl_casino_train/dpo5k_dense_tulu3/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500}"

# Downstream run ids (stable names recommended for resume)
export DPO_DENSE_RUN_ID="${DPO_DENSE_RUN_ID:-dpo5k_dense_${DPO_DS_KEY}}"
export DPO_SPARSE_RANDOM_RUN_ID="${DPO_SPARSE_RANDOM_RUN_ID:-dpo5k_sparse_random_${DPO_DS_KEY}}"
export DPO_SPARSE_CAV_RUN_ID="${DPO_SPARSE_CAV_RUN_ID:-dpo5k_sparse_cav_${DPO_DS_KEY}}"
export DPO_SPARSE_SNIP_RUN_ID="${DPO_SPARSE_SNIP_RUN_ID:-dpo5k_sparse_snip_${DPO_DS_KEY}}"

export DPO_DENSE_RUN_ID_LIGHT_R1="${DPO_DENSE_RUN_ID_LIGHT_R1:-dpo5k_dense_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1="${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1:-dpo5k_sparse_random_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_CAV_RUN_ID_LIGHT_R1="${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1:-dpo5k_sparse_cav_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1="${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1:-dpo5k_sparse_snip_${DPO_DS_KEY_LIGHT_R1}}"

# Sparse DPO with checkpoint-diff “oracle” masks (requires dense runs at ORACLE_FINAL_CKPT_DIR*).
export DPO_SPARSE_ORACLE_RUN_ID="${DPO_SPARSE_ORACLE_RUN_ID:-dpo_sparse_oracle_${DPO_DS_KEY}}"
export DPO_SPARSE_ORACLE_RUN_ID_LIGHT_R1="${DPO_SPARSE_ORACLE_RUN_ID_LIGHT_R1:-dpo_sparse_oracle_${DPO_DS_KEY_LIGHT_R1}}"
# 0 = skip checkpoint_diff_mask_finder + sparse-oracle sbatch (e.g. dense ckpts not ready yet).
export ORCH_QUEUE_ORACLE_MASK_TRAINING="${ORCH_QUEUE_ORACLE_MASK_TRAINING:-1}"

export GRPO_DENSE_RUN_SLUG="${GRPO_DENSE_RUN_SLUG:-llama31_${GRPO_DATASET_HF##*/}_grpo_dense_v1}"
export GRPO_SPARSE_RANDOM_RUN_NAME="${GRPO_SPARSE_RANDOM_RUN_NAME:-grpo_sparse_random_sp975_v1}"
export GRPO_SPARSE_CAV_RUN_NAME="${GRPO_SPARSE_CAV_RUN_NAME:-grpo_sparse_cav_sp975_v1}"
export GRPO_SPARSE_SNIP_RUN_NAME="${GRPO_SPARSE_SNIP_RUN_NAME:-grpo_sparse_snip_${GRPO_SNIP_OBJECTIVE}_sp975_v1}"
export GRPO_SPARSE_GRASP_RUN_NAME="${GRPO_SPARSE_GRASP_RUN_NAME:-grpo_sparse_grasp_${GRPO_GRASP_OBJECTIVE}_sp975_v1}"

# Downstream training knobs (DPO doc defaults)
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-500}"
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-50}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"

# Downstream training knobs (GRPO doc defaults)
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-500}"
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-2048}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"

# Queued training jobs: opt-in soft timeout + chained sbatch (see scripts/README.md).
export ORCH_USE_TRAIN_AUTO_RESUME="${ORCH_USE_TRAIN_AUTO_RESUME:-0}"
export MAX_AUTO_RESUME="${MAX_AUTO_RESUME:-8}"
export ORCH_TRAIN_PARTITION="${ORCH_TRAIN_PARTITION:-gpu}"
export ORCH_TRAIN_GRES="${ORCH_TRAIN_GRES:-gpu:h200:1}"
# Sparse training (DPO + GRPO): default A100 — override with ORCH_TRAIN_PARTITION_SPARSE / ORCH_TRAIN_GRES_SPARSE
export ORCH_TRAIN_PARTITION_SPARSE="${ORCH_TRAIN_PARTITION_SPARSE:-${ORCH_TRAIN_PARTITION:-gpu}}"
export ORCH_TRAIN_GRES_SPARSE="${ORCH_TRAIN_GRES_SPARSE:-gpu:a100:1}"
export ORCH_TRAIN_MEM="${ORCH_TRAIN_MEM:-128G}"
export ORCH_TRAIN_CPUS="${ORCH_TRAIN_CPUS:-8}"
export ORCH_TRAIN_TIME_DPO="${ORCH_TRAIN_TIME_DPO:-07:45:00}"
export ORCH_TRAIN_TIME_GRPO="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"
if [ "${ORCH_USE_TRAIN_AUTO_RESUME}" = "1" ] && [ -n "${ORCH_TRAIN_SOFT_SECONDS:-}" ]; then
  export AUTO_RESUME_SOFT_SECONDS="${ORCH_TRAIN_SOFT_SECONDS}"
fi
# When ORCH_USE_TRAIN_AUTO_RESUME=1, downstream sbatch uses orchestrate_training_child_entry.sh with
# ORCH_TRAIN_PARTITION (default gpu), ORCH_TRAIN_GRES (dense: gpu:h200:1), ORCH_TRAIN_GRES_SPARSE (default gpu:a100:1),
# ORCH_TRAIN_TIME_DPO (07:45:00), ORCH_TRAIN_TIME_GRPO (08:00:00), ORCH_TRAIN_CPUS (8 for GRPO jobs).

# DPO/GRPO training always logs to W&B (Python entrypoints + launch scripts force online).
unset WANDB_DISABLED 2>/dev/null || true
unset WANDB_SILENT 2>/dev/null || true
export WANDB_MODE="online"

########################################
# Focus mode: GRPO (Open-R1) 3-way (one-and-done)
########################################
# When enabled, this orchestrator will:
# - Generate random GRPO masks (element + block)
# - Generate GRaSP GRPO masks (element + block), with SNR reweighting enabled
# - Queue ONLY:
#     1) dense GRPO
#     2) random sparse GRPO
#     3) GRaSP sparse GRPO
# - Train for 1000 steps (GRPO_TARGET_STEPS / ORCH_GRPO_TRAIN_STEPS), then exit early (skips DPO dense/oracle/Wanda/SNIP/etc.)
# Full DPO sweep (dense Tulu3 + Light-R1, oracle sparse, Wanda, …): export ORCH_GRPO_THREEWAY_FOCUS=0 before sbatch.
export ORCH_GRPO_THREEWAY_FOCUS="${ORCH_GRPO_THREEWAY_FOCUS:-1}"
export ORCH_GRPO_TRAIN_STEPS="${ORCH_GRPO_TRAIN_STEPS:-1000}"
# Higher calibration by default for this focus mode
export ORCH_GRPO_CALIB_SAMPLES="${ORCH_GRPO_CALIB_SAMPLES:-256}"
export ORCH_GRPO_CALIB_BATCH_SIZE="${ORCH_GRPO_CALIB_BATCH_SIZE:-4}"
# Which mask variant to use for training (masks for both are generated)
export ORCH_GRPO_RANDOM_VARIANT="${ORCH_GRPO_RANDOM_VARIANT:-element}" # element|block
export ORCH_GRPO_GRASP_VARIANT="${ORCH_GRPO_GRASP_VARIANT:-element}"   # element|block
export ORCH_GRPO_MASK_BLOCK_SIZE="${ORCH_GRPO_MASK_BLOCK_SIZE:-256}"
export ORCH_GRPO_MASK_BLOCK_REDUCTION="${ORCH_GRPO_MASK_BLOCK_REDUCTION:-mean}" # mean|max

########################################
# Preflight
########################################

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
  exit 1
fi

"$TRAIN_PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in this job; masks require a GPU allocation.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: device_count={torch.cuda.device_count()} device0={torch.cuda.get_device_name(0)!r}")
PY

sanitize_model_name() {
  # Must match the repo’s sanitize conventions (see inference_mask_finder.py / pipeline_common.sh).
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

mkdir -p "$MASK_DIR" "$GRPO_MASK_DIR"
echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "SCRATCH_USER_ROOT=${SCRATCH_USER_ROOT}"
echo "MASK_DIR=${MASK_DIR}"
echo "GRPO_MASK_DIR=${GRPO_MASK_DIR}"
echo "MODEL=${MODEL} (sanitized=${MODEL_SANITIZED})"
echo "DPO_DS_KEY=${DPO_DS_KEY}  COLD_DATASET_HF=${COLD_DATASET_HF}"
echo "GRPO_DATASET_HF=${GRPO_DATASET_HF}  GRPO_DATASET=${GRPO_DATASET}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT}  MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}"
echo "DPO_SNIP_OBJECTIVE=${DPO_SNIP_OBJECTIVE}  GRPO_SNIP_OBJECTIVE=${GRPO_SNIP_OBJECTIVE}"
echo "ORCH_CAV_IMPL=${ORCH_CAV_IMPL}  ORCH_CAV_WEIGHT_ABS=${ORCH_CAV_WEIGHT_ABS}  ORCH_CAV_INCLUDE_EMBEDS=${ORCH_CAV_INCLUDE_EMBEDS}"
echo "ORCH_SCORE_SNR=${ORCH_SCORE_SNR} ORCH_SCORE_SNR_TRANSFORM=${ORCH_SCORE_SNR_TRANSFORM} ORCH_SCORE_SNR_EPS=${ORCH_SCORE_SNR_EPS}"
echo "ORCH_MASK_GRANULARITY=${ORCH_MASK_GRANULARITY} ORCH_MASK_BLOCK_SIZE=${ORCH_MASK_BLOCK_SIZE} ORCH_MASK_BLOCK_REDUCTION=${ORCH_MASK_BLOCK_REDUCTION}"
echo "ORCH_MASK_FNAME_TAG=${ORCH_MASK_FNAME_TAG:-}  (SNIP/GRaSP .pt suffix; empty = element-wise legacy names)"

########################################
# 1) Helper functions for submission
########################################

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

ORCH_TRAIN_ENTRY="${REPO_ROOT}/scripts/orchestrate_training_child_entry.sh"

_orch_train_bases_export() {
  echo "NUM_STEPS_DPO=${NUM_STEPS_DPO},GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS},TRAIN_OUT_BASE=${TRAIN_OUT_BASE},SPARSE_OUT_BASE=${SPARSE_OUT_BASE},GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE},GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE},GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS},GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
}

submit_training_job() {
  local label="$1"
  local inner_script="$2"
  local auto_mode="$3"
  local export_extra="$4"
  local job_tag="${5:-$auto_mode}"

  local bases_steps jid
  bases_steps="$(_orch_train_bases_export)"

  # NOTE: We no longer use --dependency="afterok:${SLURM_JOB_ID}" because we want to launch
  # immediately after the mask is confirmed on disk.
  if [ "${ORCH_USE_TRAIN_AUTO_RESUME:-0}" != "1" ]; then
    local part_na gres_na mem_na time_wall_na
    local -a cpus_na=()
    part_na="${ORCH_TRAIN_PARTITION:-gpu}"
    gres_na="${ORCH_TRAIN_GRES:-gpu:h200:1}"
    mem_na="${ORCH_TRAIN_MEM:-128G}"
    case "${auto_mode}" in
      sparse_dpo|sparse_grpo)
        part_na="${ORCH_TRAIN_PARTITION_SPARSE:-${ORCH_TRAIN_PARTITION:-gpu}}"
        gres_na="${ORCH_TRAIN_GRES_SPARSE:-gpu:a100:1}"
        ;;
    esac
    case "${auto_mode}" in
      dense_dpo|sparse_dpo)
        time_wall_na="${ORCH_TRAIN_TIME_DPO:-07:45:00}"
        ;;
      dense_grpo|sparse_grpo)
        time_wall_na="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"
        cpus_na=( --cpus-per-task="${ORCH_TRAIN_CPUS:-8}" )
        ;;
      *)
        echo "ERROR: unknown auto-resume mode tag: ${auto_mode}" >&2
        exit 1
        ;;
    esac
    jid=$(sbatch --parsable \
      --partition="${part_na}" --nodes=1 --ntasks=1 \
      --gres="${gres_na}" --mem="${mem_na}" --time="${time_wall_na}" \
      "${cpus_na[@]}" \
      --export=ALL,"${export_extra},${bases_steps}" \
      "${inner_script}")
    echo "${label}: ${jid}"
    return
  fi

  local part gres mem time_wall out_log err_log jname full_export
  local -a cpus_args=()
  part="${ORCH_TRAIN_PARTITION:-gpu}"
  gres="${ORCH_TRAIN_GRES:-gpu:h200:1}"
  case "${auto_mode}" in
    sparse_dpo|sparse_grpo)
      part="${ORCH_TRAIN_PARTITION_SPARSE:-${ORCH_TRAIN_PARTITION:-gpu}}"
      gres="${ORCH_TRAIN_GRES_SPARSE:-gpu:a100:1}"
      ;;
  esac
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

  jid=$(sbatch --parsable \
    --partition="${part}" --nodes=1 --ntasks=1 \
    --gres="${gres}" --mem="${mem}" --time="${time_wall}" \
    "${cpus_args[@]}" \
    --job-name="${jname}" \
    --output="${out_log}" --error="${err_log}" \
    --export="${full_export}" \
    "${ORCH_TRAIN_ENTRY}")
  echo "${label}: ${jid}"
}

########################################
# 1) Mask generation (GPU)
########################################
# Random baseline: one file reused for DPO (Tulu3), DPO (Light-R1), and GRPO sparse-random.
export RANDOM_MASK_SEED="${RANDOM_MASK_SEED:-42}"

# DPO masks — Tulu3 calibration
export MASK_RANDOM="${MASK_DIR}/random_${MODEL//\//_}_sparsity${SPARSITY_PERCENT}pct_seed${RANDOM_MASK_SEED}.pt"
export MASK_WANDA="${MASK_DIR}/wanda_owl_${MODEL_SANITIZED}_tulu3_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_ORACLE_TULU3="${MASK_DIR}/ground_truth_ckpt_diff_${MODEL_SANITIZED}_tulu3_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_SNIP="${MASK_DIR}/cold_snip_${MODEL_SANITIZED}_${DPO_DS_KEY}_sparsity${SPARSITY_PERCENT}pct_${DPO_SNIP_OBJECTIVE}${ORCH_MASK_FNAME_TAG}.pt"

# DPO masks — Light-R1 (same model/sparsity; Wanda+OWL calibrated on Light-R1)
export MASK_WANDA_LIGHT_R1="${MASK_DIR}/wanda_owl_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_SNIP_LIGHT_R1="${MASK_DIR}/cold_snip_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct_${DPO_SNIP_OBJECTIVE}${ORCH_MASK_FNAME_TAG}.pt"
export MASK_ORACLE_LIGHT_R1="${MASK_DIR}/ground_truth_ckpt_diff_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct.pt"

# GRPO: reuse MASK_RANDOM for sparse-random; Wanda+OWL + SNIP are GRPO-specific paths
export MASK_GRASP_LIGHT_R1="${MASK_DIR}/cold_grasp_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct_${GRASP_OBJECTIVE}${ORCH_MASK_FNAME_TAG}.pt"

export MASK_GRPO_RANDOM="${MASK_RANDOM}"
export MASK_GRPO_WANDA="${GRPO_MASK_DIR}/grpo_wanda_owl_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}.pt"
export MASK_GRPO_SNIP="${GRPO_MASK_DIR}/grpo_snip_${GRPO_SNIP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}${ORCH_MASK_FNAME_TAG}.pt"
export MASK_GRPO_GRASP="${GRPO_MASK_DIR}/grpo_grasp_${GRPO_GRASP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}${ORCH_MASK_FNAME_TAG}.pt"

SNRS=()
if [ "${ORCH_SCORE_SNR:-off}" != "off" ]; then
  SNRS+=( --score-snr "${ORCH_SCORE_SNR}" )
  SNRS+=( --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}" )
  SNRS+=( --score-snr-eps "${ORCH_SCORE_SNR_EPS}" )
  SNRS+=( --score-snr-clamp-min "${ORCH_SCORE_SNR_CLAMP_MIN}" )
  SNRS+=( --score-snr-clamp-max "${ORCH_SCORE_SNR_CLAMP_MAX}" )
  SNRS+=( --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}" )
  if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
    SNRS+=( --score-snr-allow-large-ram )
  fi
fi

MG=()
if [ "${ORCH_MASK_GRANULARITY:-element}" != "element" ]; then
  MG+=( --mask-granularity "${ORCH_MASK_GRANULARITY}" )
  MG+=( --mask-block-size "${ORCH_MASK_BLOCK_SIZE}" )
  MG+=( --mask-block-reduction "${ORCH_MASK_BLOCK_REDUCTION}" )
fi

echo ""
echo "=== Generating shared random mask (DPO + GRPO) → ${MASK_RANDOM}"
set +e
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --seed "${RANDOM_MASK_SEED}" \
  --output_file "${MASK_RANDOM}"
set -e

if [ -f "${MASK_RANDOM}" ]; then
  echo "--- Submitting Random Sparse jobs ---"
  submit_training_job "DPO sparse random Light-R1" \
    "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
    sparse_dpo \
    "PIPELINE_RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_RANDOM},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
    "dpo_sp_rand_lr1_${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1}"

  submit_training_job "GRPO sparse random" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
    sparse_grpo \
    "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_RANDOM},GRPO_RUN_NAME=${GRPO_SPARSE_RANDOM_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
    "grpo_sp_rand_${GRPO_SPARSE_RANDOM_RUN_NAME}"
else
  echo "WARNING: Random mask generation failed; skipping associated jobs."
fi

########################################
# Focus mode implementation (GRPO Open-R1 3-way)
########################################
if [ "${ORCH_GRPO_THREEWAY_FOCUS}" = "1" ]; then
  echo ""
  echo "=== ORCH_GRPO_THREEWAY_FOCUS=1: Open-R1 GRPO 3-way (1000 steps) ==="

  export GRPO_TARGET_STEPS="${ORCH_GRPO_TRAIN_STEPS}"
  export GRPO_N_SAMPLES="${ORCH_GRPO_CALIB_SAMPLES}"
  export ORCH_MASK_BATCH_SIZE="${ORCH_GRPO_CALIB_BATCH_SIZE}"
  export ORCH_SCORE_SNR="${ORCH_SCORE_SNR:-per_weight}"

  echo "Focus knobs: GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS} GRPO_N_SAMPLES=${GRPO_N_SAMPLES} ORCH_MASK_BATCH_SIZE=${ORCH_MASK_BATCH_SIZE}"
  echo "Mask variants: random=${ORCH_GRPO_RANDOM_VARIANT} grasp=${ORCH_GRPO_GRASP_VARIANT} block=${ORCH_GRPO_MASK_BLOCK_SIZE} reduction=${ORCH_GRPO_MASK_BLOCK_REDUCTION}"
  echo "SNR knobs: ORCH_SCORE_SNR=${ORCH_SCORE_SNR} transform=${ORCH_SCORE_SNR_TRANSFORM} budget_gb=${ORCH_SCORE_SNR_RAM_BUDGET_GB} allow_large_ram=${ORCH_SCORE_SNR_ALLOW_LARGE_RAM}"

  # Run names for this 3-way.
  export GRPO_RUN_SLUG="${GRPO_RUN_SLUG:-grpo_dense_openr1_steps${GRPO_TARGET_STEPS}_v1}"
  export GRPO_SPARSE_RANDOM_RUN_NAME="${GRPO_SPARSE_RANDOM_RUN_NAME:-grpo_sparse_random_openr1_steps${GRPO_TARGET_STEPS}_v1}"
  export GRPO_SPARSE_GRASP_RUN_NAME="${GRPO_SPARSE_GRASP_RUN_NAME:-grpo_sparse_grasp_openr1_steps${GRPO_TARGET_STEPS}_v1}"

  # Masks (write under GRPO_MASK_DIR; include variant tags so runs never overwrite).
  export MASK_GRPO_RANDOM_ELEM="${MASK_GRPO_RANDOM}"
  export MASK_GRPO_RANDOM_BLOCK="${GRPO_MASK_DIR}/grpo_random_block_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}_block${ORCH_GRPO_MASK_BLOCK_SIZE}.pt"
  export MASK_GRPO_GRASP_ELEM_SNR="${GRPO_MASK_DIR}/grpo_grasp_${GRPO_GRASP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}_elem_snr${ORCH_SCORE_SNR}_${ORCH_SCORE_SNR_TRANSFORM}.pt"
  export MASK_GRPO_GRASP_BLOCK_SNR="${GRPO_MASK_DIR}/grpo_grasp_${GRPO_GRASP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}_block${ORCH_GRPO_MASK_BLOCK_SIZE}_${ORCH_GRPO_MASK_BLOCK_REDUCTION}_snr${ORCH_SCORE_SNR}_${ORCH_SCORE_SNR_TRANSFORM}.pt"

  # Generate a block-structured random mask (block grid selection then expand) for Open-R1.
  echo ""
  echo "=== Generating GRPO random BLOCK mask → ${MASK_GRPO_RANDOM_BLOCK}"
  "$TRAIN_PY" - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM
from src.utils.mask_utils import create_mask_from_scores_gpu_efficient, save_masks

model_name = os.environ["MODEL"]
out_path = os.environ["MASK_GRPO_RANDOM_BLOCK"]
sparsity = float(os.environ["SPARSITY_PERCENT"])
min_keep = float(os.environ["MIN_LAYER_KEEP_RATIO"])
block = int(os.environ["ORCH_GRPO_MASK_BLOCK_SIZE"])
seed = int(os.environ.get("RANDOM_MASK_SEED", "42"))
torch.manual_seed(seed)

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="meta",
    low_cpu_mem_usage=True,
)
block_scores = {}
shapes = {}
for name, p in m.named_parameters():
    if "weight" not in name or p.dim() != 2:
        continue
    M, N = int(p.shape[0]), int(p.shape[1])
    shapes[name] = (M, N)
    bm = (M + block - 1) // block
    bn = (N + block - 1) // block
    block_scores[name] = torch.rand((bm, bn), dtype=torch.float32, device="cpu")

block_masks = create_mask_from_scores_gpu_efficient(
    block_scores,
    sparsity_percent=sparsity,
    device="cpu",
    local_pool=False,
    min_layer_keep_ratio=min_keep,
    add_tie_break_noise=False,
)
final = {}
for name, b in block_masks.items():
    M, N = shapes[name]
    expanded = b.bool().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    final[name] = expanded[:M, :N].to(dtype=torch.bool)

meta = {
    "method": "random_block_global",
    "model_name": model_name,
    "sparsity_percent": sparsity,
    "block_size": block,
    "seed": seed,
    "min_layer_keep_ratio": min_keep,
}
save_masks(final, out_path, metadata=meta)
print(f"Wrote block-random mask: {out_path}")
PY

  # Generate GRaSP masks (element+block), both with SNR per_weight.
  SNRS_ON=( --score-snr "${ORCH_SCORE_SNR}" )
  SNRS_ON+=( --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}" )
  SNRS_ON+=( --score-snr-eps "${ORCH_SCORE_SNR_EPS}" )
  SNRS_ON+=( --score-snr-clamp-min "${ORCH_SCORE_SNR_CLAMP_MIN}" )
  SNRS_ON+=( --score-snr-clamp-max "${ORCH_SCORE_SNR_CLAMP_MAX}" )
  SNRS_ON+=( --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}" )
  if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
    SNRS_ON+=( --score-snr-allow-large-ram )
  fi

  echo ""
  echo "=== Generating GRPO GRaSP element+SNR → ${MASK_GRPO_GRASP_ELEM_SNR}"
  "$TRAIN_PY" src/cold_start/inference_mask_finder.py \
    --model_name "${MODEL}" \
    --mode grpo \
    --method grasp \
    --grasp-objective "${GRPO_GRASP_OBJECTIVE}" \
    --dataset_name "${GRPO_DATASET_HF}" \
    --n_samples "${GRPO_N_SAMPLES}" \
    --sparsity "${SPARSITY_PERCENT}" \
    --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
    --batch_size "${ORCH_MASK_BATCH_SIZE}" \
    --seed 42 \
    --grasp-preference-beta "${GRPO_GRASP_PREFERENCE_BETA}" \
    "${SNRS_ON[@]}" \
    --mask-granularity element \
    --output "${MASK_GRPO_GRASP_ELEM_SNR}"

  echo ""
  echo "=== Generating GRPO GRaSP block+SNR → ${MASK_GRPO_GRASP_BLOCK_SNR}"
  "$TRAIN_PY" src/cold_start/inference_mask_finder.py \
    --model_name "${MODEL}" \
    --mode grpo \
    --method grasp \
    --grasp-objective "${GRPO_GRASP_OBJECTIVE}" \
    --dataset_name "${GRPO_DATASET_HF}" \
    --n_samples "${GRPO_N_SAMPLES}" \
    --sparsity "${SPARSITY_PERCENT}" \
    --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
    --batch_size "${ORCH_MASK_BATCH_SIZE}" \
    --seed 42 \
    --grasp-preference-beta "${GRPO_GRASP_PREFERENCE_BETA}" \
    "${SNRS_ON[@]}" \
    --mask-granularity block \
    --mask-block-size "${ORCH_GRPO_MASK_BLOCK_SIZE}" \
    --mask-block-reduction "${ORCH_GRPO_MASK_BLOCK_REDUCTION}" \
    --output "${MASK_GRPO_GRASP_BLOCK_SNR}"

  # Select which masks to use for the sparse jobs.
  case "${ORCH_GRPO_RANDOM_VARIANT}" in
    block) export MASK_GRPO_RANDOM_TRAIN="${MASK_GRPO_RANDOM_BLOCK}" ;;
    *)     export MASK_GRPO_RANDOM_TRAIN="${MASK_GRPO_RANDOM_ELEM}" ;;
  esac
  case "${ORCH_GRPO_GRASP_VARIANT}" in
    block) export MASK_GRPO_GRASP_TRAIN="${MASK_GRPO_GRASP_BLOCK_SNR}" ;;
    *)     export MASK_GRPO_GRASP_TRAIN="${MASK_GRPO_GRASP_ELEM_SNR}" ;;
  esac

  echo ""
  echo "--- Submitting GRPO jobs (dense, random sparse, GRaSP sparse) ---"
  submit_training_job "GRPO dense (1000 steps)" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
    dense_grpo \
    "GRPO_MODE=dense,GRPO_RUN_SLUG=${GRPO_RUN_SLUG},GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
    "grpo_dense_${GRPO_RUN_SLUG}"

  submit_training_job "GRPO sparse random (1000 steps)" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
    sparse_grpo \
    "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_RANDOM_TRAIN},GRPO_RUN_NAME=${GRPO_SPARSE_RANDOM_RUN_NAME},GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
    "grpo_sp_rand_${GRPO_SPARSE_RANDOM_RUN_NAME}"

  submit_training_job "GRPO sparse GRaSP (1000 steps)" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
    sparse_grpo \
    "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_GRASP_TRAIN},GRPO_RUN_NAME=${GRPO_SPARSE_GRASP_RUN_NAME},GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
    "grpo_sp_grasp_${GRPO_SPARSE_GRASP_RUN_NAME}"

  echo ""
  echo "=== GRPO 3-way focus complete (queued requested jobs). Exiting early. ==="
  echo "Mask dirs: MASK_DIR=${MASK_DIR} GRPO_MASK_DIR=${GRPO_MASK_DIR}"
  echo "Monitor: squeue -u ${USER:-<user>}"
  exit 0
fi

echo ""
echo "--- Submitting Dense jobs (no mask needed) ---"
submit_training_job "DPO dense ${DPO_DS_KEY}" \
  "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh" \
  dense_dpo \
  "PIPELINE_CHAIN_NEXT_STAGE=0,PIPELINE_RUN_ID=${DPO_DENSE_RUN_ID},RUN_ID=${DPO_DENSE_RUN_ID},DPO_DATASET_KEY=${DPO_DS_KEY}" \
  "dpo_dense_${DPO_DENSE_RUN_ID}"

submit_training_job "DPO dense ${DPO_DS_KEY_LIGHT_R1}" \
  "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh" \
  dense_dpo \
  "PIPELINE_CHAIN_NEXT_STAGE=0,PIPELINE_RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1},RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
  "dpo_dense_lr1_${DPO_DENSE_RUN_ID_LIGHT_R1}"

if [ "${ORCH_QUEUE_ORACLE_MASK_TRAINING:-1}" = "1" ]; then
  echo ""
  echo "=== Oracle masks (|θ_final−θ_init|) via checkpoint_diff_mask_finder → sparse DPO ==="
  echo "Tulu3 final dir:  ${ORACLE_FINAL_CKPT_DIR_TULU3}"
  echo "Light-R1 final dir: ${ORACLE_FINAL_CKPT_DIR}"

  mkdir -p "${MASK_DIR}"

  if [ -d "${ORACLE_FINAL_CKPT_DIR_TULU3}" ]; then
    echo "--- Building oracle mask (Tulu3) → ${MASK_ORACLE_TULU3}"
    set +e
    "$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
      --initial_model "${MODEL}" \
      --final_model "${ORACLE_FINAL_CKPT_DIR_TULU3}" \
      --sparsity_percent "${SPARSITY_PERCENT}" \
      --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
      --output_file "${MASK_ORACLE_TULU3}"
    _oracle_t_rc=$?
    set -e
    if [ "${_oracle_t_rc}" -ne 0 ] || [ ! -f "${MASK_ORACLE_TULU3}" ]; then
      echo "WARNING: Tulu3 oracle mask failed or missing (rc=${_oracle_t_rc}); skip sparse oracle Tulu3."
    else
      submit_training_job "DPO sparse oracle ${DPO_DS_KEY}" \
        "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
        sparse_dpo \
        "PIPELINE_RUN_ID=${DPO_SPARSE_ORACLE_RUN_ID},RUN_ID=${DPO_SPARSE_ORACLE_RUN_ID},PIPELINE_MASK_FILE=${MASK_ORACLE_TULU3},DPO_DATASET_KEY=${DPO_DS_KEY}" \
        "dpo_sp_oracle_${DPO_SPARSE_ORACLE_RUN_ID}"
    fi
  else
    echo "WARNING: ORACLE_FINAL_CKPT_DIR_TULU3 not a directory — skip Tulu3 oracle mask/sparse (${ORACLE_FINAL_CKPT_DIR_TULU3})."
  fi

  if [ -d "${ORACLE_FINAL_CKPT_DIR}" ]; then
    echo "--- Building oracle mask (Light-R1) → ${MASK_ORACLE_LIGHT_R1}"
    set +e
    "$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
      --initial_model "${MODEL}" \
      --final_model "${ORACLE_FINAL_CKPT_DIR}" \
      --sparsity_percent "${SPARSITY_PERCENT}" \
      --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
      --output_file "${MASK_ORACLE_LIGHT_R1}"
    _oracle_lr_rc=$?
    set -e
    if [ "${_oracle_lr_rc}" -ne 0 ] || [ ! -f "${MASK_ORACLE_LIGHT_R1}" ]; then
      echo "WARNING: Light-R1 oracle mask failed or missing (rc=${_oracle_lr_rc}); skip sparse oracle Light-R1."
    else
      submit_training_job "DPO sparse oracle ${DPO_DS_KEY_LIGHT_R1}" \
        "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
        sparse_dpo \
        "PIPELINE_RUN_ID=${DPO_SPARSE_ORACLE_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_ORACLE_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_ORACLE_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
        "dpo_sp_oracle_lr1_${DPO_SPARSE_ORACLE_RUN_ID_LIGHT_R1}"
    fi
  else
    echo "WARNING: ORACLE_FINAL_CKPT_DIR not a directory — skip Light-R1 oracle mask/sparse (${ORACLE_FINAL_CKPT_DIR})."
  fi
fi

# Light-R1 Wanda+OWL
echo ""
echo "=== Generating DPO Wanda+OWL mask (Light-R1) → ${MASK_WANDA_LIGHT_R1}"
set +e
"$TRAIN_PY" src/cold_start/wanda_owl_mask_finder.py \
  --model_name "${MODEL}" \
  --method wanda_owl \
  --n_samples "${COLD_CAV_SUBSET}" \
  --max_length "${DPO_MAX_LENGTH}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --include_embeddings \
  --output_file "${MASK_WANDA_LIGHT_R1}"
set -e

if [ -f "${MASK_WANDA_LIGHT_R1}" ]; then
  submit_training_job "DPO sparse Wanda+OWL Light-R1" \
    "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
    sparse_dpo \
    "PIPELINE_RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_WANDA_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
    "dpo_sp_wanda_lr1_${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}"
fi

# Light-R1 SNIP
echo ""
echo "=== Generating DPO SNIP mask (Light-R1) → ${MASK_SNIP_LIGHT_R1}"
set +e
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --method snip \
  --mode dpo \
  --dataset_name "${DPO_DS_KEY_LIGHT_R1}" \
  --snip-objective "${DPO_SNIP_OBJECTIVE}" \
  --n_samples "${COLD_CAV_SUBSET}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --max_length "${DPO_MAX_LENGTH}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
  --snip-preference-beta "${DPO_BETA}" \
  "${SNRS[@]}" \
  "${MG[@]}" \
  --output "${MASK_SNIP_LIGHT_R1}"
set -e

if [ -f "${MASK_SNIP_LIGHT_R1}" ]; then
  submit_training_job "DPO sparse SNIP Light-R1" \
    "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
    sparse_dpo \
    "PIPELINE_RUN_ID=${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_SNIP_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
    "dpo_sp_snip_lr1_${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1}"
fi

# Light-R1 GRaSP
echo ""
echo "=== Generating DPO GRaSP mask (Light-R1) → ${MASK_GRASP_LIGHT_R1}"
set +e
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --method grasp \
  --mode dpo \
  --dataset_name "${DPO_DS_KEY_LIGHT_R1}" \
  --grasp-objective "${GRASP_OBJECTIVE}" \
  --n_samples "${COLD_CAV_SUBSET}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --batch_size 1 \
  --max_length "${DPO_MAX_LENGTH}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  --grasp-preference-beta "${DPO_BETA}" \
  "${SNRS[@]}" \
  "${MG[@]}" \
  --output "${MASK_GRASP_LIGHT_R1}"
set -e

if [ -f "${MASK_GRASP_LIGHT_R1}" ]; then
  submit_training_job "DPO sparse GRaSP Light-R1" \
    "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
    sparse_dpo \
    "PIPELINE_RUN_ID=dpo5k_sparse_grasp_lr1,RUN_ID=dpo5k_sparse_grasp_lr1,PIPELINE_MASK_FILE=${MASK_GRASP_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
    "dpo_sp_grasp_lr1"
fi

# GRPO Wanda+OWL
echo ""
echo "=== Generating GRPO Wanda+OWL mask → ${MASK_GRPO_WANDA}"
set +e
"$TRAIN_PY" src/cold_start/wanda_owl_mask_finder.py \
  --model_name "${MODEL}" \
  --method wanda_owl \
  --n_samples "${GRPO_N_SAMPLES}" \
  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --include_embeddings \
  --output_file "${MASK_GRPO_WANDA}"
set -e

if [ -f "${MASK_GRPO_WANDA}" ]; then
  submit_training_job "GRPO sparse Wanda+OWL" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
    sparse_grpo \
    "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_WANDA},GRPO_RUN_NAME=${GRPO_SPARSE_CAV_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
    "grpo_sp_wanda_${GRPO_SPARSE_CAV_RUN_NAME}"
fi

# GRPO SNIP
echo ""
echo "=== Generating GRPO SNIP mask (${GRPO_SNIP_OBJECTIVE}) → ${MASK_GRPO_SNIP}"
set +e
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method snip \
  --snip-objective "${GRPO_SNIP_OBJECTIVE}" \
  --dataset_name "${GRPO_DATASET_HF}" \
  --n_samples "${GRPO_N_SAMPLES}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --seed 42 \
  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
  --snip-preference-beta "${GRPO_SNIP_PREFERENCE_BETA}" \
  "${SNRS[@]}" \
  "${MG[@]}" \
  --output "${MASK_GRPO_SNIP}"
set -e

if [ -f "${MASK_GRPO_SNIP}" ]; then
  submit_training_job "GRPO sparse SNIP" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
    sparse_grpo \
    "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_SNIP},GRPO_RUN_NAME=${GRPO_SPARSE_SNIP_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
    "grpo_sp_snip_${GRPO_SPARSE_SNIP_RUN_NAME}"
fi

# GRPO GRaSP
echo ""
echo "=== Generating GRPO GRaSP mask (${GRPO_GRASP_OBJECTIVE}) → ${MASK_GRPO_GRASP}"
set +e
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method grasp \
  --grasp-objective "${GRPO_GRASP_OBJECTIVE}" \
  --dataset_name "${GRPO_DATASET_HF}" \
  --n_samples "${GRPO_N_SAMPLES}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
  --batch_size 1 \
  --seed 42 \
  --grasp-preference-beta "${GRPO_GRASP_PREFERENCE_BETA}" \
  "${SNRS[@]}" \
  "${MG[@]}" \
  --output "${MASK_GRPO_GRASP}"
set -e

if [ -f "${MASK_GRPO_GRASP}" ]; then
  submit_training_job "GRPO sparse GRaSP" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
    sparse_grpo \
    "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_GRASP},GRPO_RUN_NAME=${GRPO_SPARSE_GRASP_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}" \
    "grpo_sp_grasp_${GRPO_SPARSE_GRASP_RUN_NAME}"
fi

echo ""
echo "=== Mask outputs written ==="
ls -lh "${MASK_DIR}"
echo "(Listing all generated masks in ${MASK_DIR})"

echo ""
echo "=== All attempted generation steps complete. ==="
echo "Monitor with: squeue -u ${USER:-<user>}"

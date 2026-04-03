#!/bin/bash
# Multi-GPU Stage 1: dense DPO (+ warm-start delta logs).
#
# This is a parallel entrypoint to scripts/pipeline_stage_01_dense.sh, but launches
# training with torchrun (1 process per GPU).
#
# Reservation use (optional): set MULTIGPU_RESERVATION and adjust Slurm header lines if needed.
#
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --job-name=pipe_p1_dense_mgpu
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p1_dense_mgpu.out
#SBATCH --error=logs/pipeline_%j_p1_dense_mgpu.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/../.." && pwd)"
fi
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/multigpu_pipeline/pipeline_common_multigpu.sh"
pipeline_setup_multigpu

echo "===== MULTIGPU STAGE 1/5: dense DPO (${RUN_ID}) ====="

echo "NOTE: If you are using a reservation, add lines like:"
echo "  #SBATCH -p reservation"
echo "  #SBATCH --reservation=<reservation_name>"
echo "and set MULTIGPU_NGPUS to match --gres."

ds="${DPO_DATASET_KEY}"
out_base="${TRAIN_OUT_BASE}/${RUN_ID}"
cache_dir="${HF_DATASETS_CACHE_ROOT:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
run_name="fullpipe_dpo_multigpu_${ds}_${RUN_ID}"
mkdir -p "$out_base"

subset_args=()
if [ -n "${SUBSET_DPO:-}" ]; then
  subset_args+=(--subset_size "$SUBSET_DPO")
fi

delta_end_args=()
if [ -n "${DELTA_LOG_END_STEP:-}" ]; then
  delta_end_args+=(--delta_log_end_step "$DELTA_LOG_END_STEP")
fi

# Resolve actual visible GPU count (prevents torchrun ranks > visible devices).
visible_gpus_env="${CUDA_VISIBLE_DEVICES:-}"
if [ -n "${visible_gpus_env}" ]; then
  IFS=',' read -r -a _gpu_arr <<< "${visible_gpus_env}"
  ngpus_visible="${#_gpu_arr[@]}"
else
  ngpus_visible="$("$TRAIN_PY" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
fi
ngpus_visible="${ngpus_visible:-0}"

if [ "${MULTIGPU_NGPUS}" = "auto" ]; then
  MULTIGPU_NGPUS="${ngpus_visible}"
elif [ "${ngpus_visible}" != "0" ] && [ "${MULTIGPU_NGPUS}" -gt "${ngpus_visible}" ]; then
  echo "ERROR: MULTIGPU_NGPUS=${MULTIGPU_NGPUS} but only ${ngpus_visible} GPU(s) are visible (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}')." >&2
  echo "Fix: request more GPUs in sbatch (--gres) or lower MULTIGPU_NGPUS." >&2
  exit 1
fi

if [ "${MULTIGPU_NGPUS}" -le 0 ]; then
  echo "ERROR: No GPUs visible. torch.cuda.device_count()=${ngpus_visible} CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'" >&2
  exit 1
fi

echo "Visible GPUs: ${ngpus_visible} | Using MULTIGPU_NGPUS=${MULTIGPU_NGPUS}"

# Effective batch math check (for logs + WandB config clarity)
eff_batch=$(( PER_DEVICE_BS * GRAD_ACCUM * MULTIGPU_NGPUS ))
echo "Effective batch = ${PER_DEVICE_BS} (per-device) * ${GRAD_ACCUM} (grad_accum) * ${MULTIGPU_NGPUS} (world) = ${eff_batch}"

train_len_args=()
if [ -n "${NUM_EPOCHS:-}" ]; then
  train_len_args+=(--num_train_epochs "${NUM_EPOCHS}")
elif [ -n "${NUM_STEPS_DPO:-}" ]; then
  train_len_args+=(--num_steps "${NUM_STEPS_DPO}")
else
  echo "ERROR: set either NUM_EPOCHS or NUM_STEPS_DPO for dense DPO length." >&2
  exit 1
fi

timeout --signal=TERM --kill-after=60 "${TRAIN_TIMEOUT_PER_DATASET}" \
  "${TRAIN_ENV}/bin/torchrun" --standalone --nproc_per_node="${MULTIGPU_NGPUS}" \
    src/full_training/DPO_train.py \
      --model_name "$MODEL" \
      --dataset "$ds" \
      "${train_len_args[@]}" \
      --per_device_train_batch_size "${PER_DEVICE_BS}" \
      --gradient_accumulation_steps "${GRAD_ACCUM}" \
      --learning_rate "${LR_PEAK}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --max_length "${SEQ_LEN}" \
      --max_prompt_length "${SEQ_LEN}" \
      --delta_log_interval "${DELTA_LOG_INTERVAL}" \
      "${delta_end_args[@]}" \
      "${subset_args[@]}" \
      --output_base_dir "$out_base" \
      --dataset_cache_dir "$cache_dir" \
      --use_wandb \
      --run_name "$run_name" 2>&1 | tee "logs/full_pipeline_dpo_multigpu_${ds}_${RUN_ID}.log"

echo "===== MULTIGPU STAGE 1 COMPLETE (${RUN_ID}) ====="
echo "Next: run masks with scripts/pipeline_stage_02_masks.sh (single-GPU) or add a multigpu mask stage wrapper later."


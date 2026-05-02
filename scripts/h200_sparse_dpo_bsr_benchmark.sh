#!/bin/bash
# Single H200 job: optional dense baseline + random-mask sparse BSR-DPO phases
# (see src/full_training/h200_sparse_dpo_bsr_benchmark.py). Default skips dense baseline for faster sweeps.
# Submit from repo root:  sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
#
# --- Why end-to-end [throughput] often differs only ~few % across phases ---
# 1) Sparse phases use Triton for sparse grad_w; this benchmark pins RL_CASINO_BSR_GRAD_INPUT_MODE to
#    **dense** (standard grad_output @ weight for grad_input). Only the *forward* matmul is dense F.linear.
#    The dense baseline phase skips injection entirely (standard nn.Linear + dense optimizer).
# 2) A Llama-style step is dominated by attention/Flash, norms, activations, DPO/ref forward paths,
#    data loading, and (with GC) extra recomputation — not only MLP/linear matmuls.
# 3) benchmark_theory.json FLOP proxies count masked linear backward math; they explicitly do NOT
#    include dense forward FLOPs (src/utils/bsr_theory_metrics.py). Do not expect wall time to track
#    theory_* columns linearly.
# 4) Dense baseline uses DPO_OPTIM (default adamw_8bit); sparse uses SparseAdamW — different optim
#    kernels; for apples-to-apples sparse vs dense *optimizer* cost, try DPO_OPTIM=adamw on the host.
# 5) Phases block_sparse_*_block1d vs *_block2d differ only in RL_CASINO_ADAM_KERNEL — tiny slice of
#    step time vs attention; small deltas between them are expected.
# 6) To unlock larger gaps: microbench isolated SparseLinear backward (parity script), torch.profiler
#    / Nsight on one step, optional --mlp_only driver flag to narrow scope, or future sparse forward.
#
# Northeastern Explorer-style defaults; override env vars as needed.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=h200_bsr_bench
#SBATCH --output=logs/h200_bsr_bench_%j.out
#SBATCH --error=logs/h200_bsr_bench_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Archive-friendly baseline line (also printed below): commit + Slurm job id + scratch output root.
echo "GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
# If H200_BSR_OUT was exported before SCRATCH_USER_ROOT existed, it becomes
# /rl_casino_h200_bsr/... (under filesystem root) → mkdir permission denied. Recompute.
case "${H200_BSR_OUT:-}" in
  /rl_casino_h200_bsr/*)
    echo "WARNING: Ignoring invalid H200_BSR_OUT=${H200_BSR_OUT} (export SCRATCH_USER_ROOT before H200_BSR_OUT on the login node)." >&2
    unset H200_BSR_OUT
    ;;
esac
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Triton: persistent compile cache on scratch (fewer recompiles across steps/phases than default TMP).
# First kernel-adjacent lever before any custom Triton tuning; see scripts/README.md (H200 BSR section).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_USER_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

# Fewer lines during ``replace_linear_modules`` (Slurm .out / NFS); set to 0 for per-layer debug prints.
export RL_CASINO_BSR_QUIET_INJECTION="${RL_CASINO_BSR_QUIET_INJECTION:-1}"

# Disable W&B / external experiment trackers (must disable console capture — see errno 116 on Slurm).
# Use fixed assignments so a stray login-node export cannot re-enable wandb stdout wrapping.
export WANDB_MODE="disabled"
export WANDB_DISABLED="true"
export WANDB_CONSOLE="off"
export WANDB_SILENT="true"

# HuggingFace Trainer tqdm → stdout; can still trip NFS errno 116. Default off for this job.
export RL_CASINO_DISABLE_TQDM="${RL_CASINO_DISABLE_TQDM:-1}"

# --- Model / data (override before sbatch or via #SBATCH --export) ---
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE_ROOT:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_DATASETS_CACHE_ROOT}"

# Steps **per phase** (default 50). Uses H200_BSR_STEPS_PER_PHASE only — we do NOT read
# NUM_STEPS_DPO here, so a leftover export NUM_STEPS_DPO=500 from other README snippets
# cannot silently turn this into a 500-step-per-phase run.
export H200_BSR_STEPS_PER_PHASE="${H200_BSR_STEPS_PER_PHASE:-8}"
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"

# Match pipeline dense `DPO_train.py` default optimizer (`--optim adamw_8bit`) when bitsandbytes is installed.
export DPO_OPTIM="${DPO_OPTIM:-adamw_8bit}"

# For stable benchmarking (avoid layer-dependent kernel variants)
# - BSR_USE_ATOMIC: keep constexpr stable across all layers in a phase
# - BSR_BATCH_CHUNKS: keep grid shape deterministic (disable auto heuristic)
export BSR_USE_ATOMIC="${BSR_USE_ATOMIC:-0}"
export BSR_BATCH_CHUNKS="${BSR_BATCH_CHUNKS:-8}"

# Grad_input for BSR backward: **dense** only for this job (no Triton sparse grad_i path).
# The driver passes --phase_grad_input_modes dense; this export is the fallback if anything reads env early.
export RL_CASINO_BSR_GRAD_INPUT_MODE="${RL_CASINO_BSR_GRAD_INPUT_MODE:-dense}"

# Sparse grid sparsity targets (comma-separated; no default 99.75% unless you export it explicitly).
# Each level: element mask vs block mask × Adam block_1d vs block_2d = 4 sparse phases.
export BENCHMARK_SPARSITIES="${BENCHMARK_SPARSITIES:-97.5,95,90}"

# 1 (default): pass --no_dense_baseline — dense AdamW phase omitted. Set to 0 to prepend one dense phase.
export H200_BSR_SKIP_DENSE="${H200_BSR_SKIP_DENSE:-1}"
SKIP_DENSE_ARGS=()
if [ "${H200_BSR_SKIP_DENSE}" != "0" ]; then
  SKIP_DENSE_ARGS+=(--no_dense_baseline)
fi

# Per-micro-batch CUDA timing + synchronize() for CSV columns t_* — default OFF (large gradient_accum
# makes this path ruin throughput). Set RL_CASINO_BSR_DETAILED_TIMING=1 only for short debug phases.
export RL_CASINO_BSR_DETAILED_TIMING="${RL_CASINO_BSR_DETAILED_TIMING:-0}"

# Trainer CSV rows follow logging frequency (default: every 25 steps). For short phases, set
# RL_CASINO_LOGGING_STEPS=1 before sbatch so each step appears in benchmark_training_log.csv.
export RL_CASINO_LOGGING_STEPS="${RL_CASINO_LOGGING_STEPS:-1}"
# If caller overrides RL_CASINO_LOGGING_STEPS, ensure it isn't larger than the phase length.
if [ "${RL_CASINO_LOGGING_STEPS}" -gt "${H200_BSR_STEPS_PER_PHASE}" ]; then
  echo "WARNING: RL_CASINO_LOGGING_STEPS=${RL_CASINO_LOGGING_STEPS} > H200_BSR_STEPS_PER_PHASE=${H200_BSR_STEPS_PER_PHASE}; clamping to 1." >&2
  export RL_CASINO_LOGGING_STEPS=1
fi

OUT_BASE="${H200_BSR_OUT:-${SCRATCH_USER_ROOT}/rl_casino_h200_bsr}/${RUN_ID:-${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_BASE"

echo "REPO_ROOT=${REPO_ROOT}"
echo "OUT_BASE=${OUT_BASE}"
echo "MODEL=${MODEL}"
echo "H200_BSR_STEPS_PER_PHASE=${H200_BSR_STEPS_PER_PHASE} (optimizer steps per dense/sparse phase)"
echo "DPO_OPTIM=${DPO_OPTIM} (dense phase; sparse uses SparseAdamW)"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}  RL_CASINO_BSR_QUIET_INJECTION=${RL_CASINO_BSR_QUIET_INJECTION}"
echo "RL_CASINO_BSR_DETAILED_TIMING=${RL_CASINO_BSR_DETAILED_TIMING}  RL_CASINO_BSR_GRAD_INPUT_MODE=${RL_CASINO_BSR_GRAD_INPUT_MODE}  RL_CASINO_LOGGING_STEPS=${RL_CASINO_LOGGING_STEPS}"
echo "H200_BSR_SKIP_DENSE=${H200_BSR_SKIP_DENSE}  (≠0 ⇒ --no_dense_baseline)"
echo "BENCHMARK_SPARSITIES=${BENCHMARK_SPARSITIES}  (each level → +4 sparse phases: elem|block × block_1d|block_2d; grad_input=dense only)"

GC_ARGS=()
if [ "${DPO_GRADIENT_CHECKPOINTING:-1}" = "0" ]; then
  GC_ARGS+=(--no_gradient_checkpointing)
fi

exec "$TRAIN_PY" src/full_training/h200_sparse_dpo_bsr_benchmark.py \
  --model_name "$MODEL" \
  --dataset tulu3 \
  --n_steps "$H200_BSR_STEPS_PER_PHASE" \
  --batch_size "$DPO_PER_DEVICE_TRAIN_BATCH_SIZE" \
  --grad_accum "$DPO_GRADIENT_ACCUMULATION_STEPS" \
  --lr "$DPO_LEARNING_RATE" \
  --warmup_ratio "$DPO_WARMUP_RATIO" \
  --max_length "$DPO_MAX_LENGTH" \
  --max_prompt_length "$DPO_MAX_PROMPT_LENGTH" \
  --output_dir "$OUT_BASE" \
  --dataset_cache_dir "$HF_DATASETS_CACHE" \
  --device_map none \
  --run_label "h200_bsr_${SLURM_JOB_ID:-local}" \
  --benchmark_sparsities "$BENCHMARK_SPARSITIES" \
  --phase_grad_input_modes "dense" \
  --phase_adam_kernels "block_1d,block_2d" \
  "${SKIP_DENSE_ARGS[@]}" \
  "${GC_ARGS[@]}"

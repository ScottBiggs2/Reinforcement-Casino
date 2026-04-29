#!/bin/bash
# Toy job: 1–2 training steps with torch.profiler and Chrome trace export.
# Avoids Nsight Systems (nsys) dependency.
#
# Submit from repo root:
#   sbatch scripts/toy_torch_profiler_bsr.sh
#
# Outputs under OUT_BASE:
# - trace.json
# - trace_summary.txt (top ops by CUDA time)

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --job-name=toy_bsr_profiler
#SBATCH --output=logs/toy_bsr_profiler_%j.out
#SBATCH --error=logs/toy_bsr_profiler_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
OUT_BASE="${OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_toys}/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_BASE"
export OUT_BASE

TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Keep compile cache stable across runs.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_USER_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
export MODEL HF_DATASETS_CACHE

# Force very frequent logging so our CSV reflects each step.
export RL_CASINO_LOGGING_STEPS="${RL_CASINO_LOGGING_STEPS:-1}"
export H200_BSR_STEPS_PER_PHASE="${H200_BSR_STEPS_PER_PHASE:-2}"

echo "OUT_BASE=${OUT_BASE}"
echo "MODEL=${MODEL}"
echo "H200_BSR_STEPS_PER_PHASE=${H200_BSR_STEPS_PER_PHASE} RL_CASINO_LOGGING_STEPS=${RL_CASINO_LOGGING_STEPS}"

"$TRAIN_PY" - <<'PY' | tee "${OUT_BASE}/trace_summary.txt"
import os, json, time
import torch
from torch.profiler import profile, ProfilerActivity

from src.full_training.h200_sparse_dpo_bsr_benchmark import main as bench_main

out = os.environ["OUT_BASE"]
trace_path = os.path.join(out, "trace.json")

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

# Run the benchmark driver in-process so profiler captures kernels.
with profile(
    activities=activities,
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
) as prof:
    bench_main()

prof.export_chrome_trace(trace_path)
print(f"✓ wrote chrome trace: {trace_path}")

try:
    # Print top CUDA ops by self time.
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
except Exception as exc:
    print(f"Could not print CUDA table: {exc}")
PY

echo "DONE. OUT_BASE=${OUT_BASE}"

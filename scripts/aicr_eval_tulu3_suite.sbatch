#!/bin/bash
# AICR port of eval_tulu3_suite.sbatch: one Tulu3-table candidate through the
# full 7-benchmark suite (mmlu, math, gsm8k, coding, ifeval, squad, gpqa_diamond)
# via src/evaluation/run_all_benchmarks.py.
#
# Submit (from anywhere on AICR):
#   MODEL_PATH=/scratch/.../final_model TAG=warm_mag sbatch scripts/aicr_eval_tulu3_suite.sbatch
# LoRA adapter instead of a full model:
#   ADAPTER=/scratch/.../checkpoint-500 TAG=lora_r64 sbatch scripts/aicr_eval_tulu3_suite.sbatch
#   (merges the adapter into the base with peft, evals the merged copy)
#
# Differences vs the Explorer original, all deliberate:
#   - b200-batch / gpu:b200:1 (homogeneous, 178 GiB)
#   - eval env is envs/rl_casino_eval (modern vllm for sm_100); training env's
#     torch 2.4/vllm 0.6.3 pins in eval_requirements.txt do NOT support Blackwell
#   - HF offline flags from aicr_env.sh are unset: benchmark datasets download
#     on first use into the shared /scratch hf_cache (network verified via curl)
#   - results go to /scratch (scratch-storage rule), not the repo's results/
#SBATCH --job-name=eval_tulu3
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=08:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/eval_tulu3_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

W=/work/neu/p2026_0038_neu/xie_yiyi
conda deactivate || true
conda activate "$W/envs/rl_casino_eval"
# Compute nodes ship an old /lib64/libstdc++ (no CXXABI_1.3.15); vLLM's spawned
# engine-core resolves it before the env's own copy and dies importing sqlite3
# (via libicui18n). Put the env's lib dir first. (2026-07-31, jobs 246429-37)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Compute nodes have no /usr/local/cuda; vLLM 0.26 on sm_100 JIT-compiles
# flashinfer's trtllm-gen fmha and needs nvcc. Use the shared cuda module's
# install directly (module cmd not guaranteed in batch shells). (jobs 246501-09)
CUDA_ROOT=/apps/aicr/packages/cuda/13.1.1/extrybz
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"

# Datasets are prefetched on the login node (scripts/aicr_prefetch_eval_datasets.py)
# and jobs run fully offline: 9 concurrent jobs racing to download into the shared
# /scratch hf_cache deadlocked on datasets' filelocks over autofs — 7h+ hangs at
# benchmark start, all of round 3 (246641-49) TIMEOUT. EVAL_ONLINE=1 restores the
# old fetch-at-runtime behavior for a single-job run.
if [ "${EVAL_ONLINE:-0}" = "1" ]; then
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_EVALUATE_OFFLINE
else
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
fi
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

TAG="${TAG:?Set TAG=<row name>}"
S=/scratch/xie_yiyi_neu
OUT_ROOT="${OUT_ROOT:-$S/rebuttal_analysis/eval_tulu3}"
# OUTPUT_DIR override lets a partial rerun write next to an earlier job's results
# so each arm keeps a single results dir; BENCHMARKS selects a subset of the suite.
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/${TAG}_${SLURM_JOB_ID}}"
BENCHMARKS="${BENCHMARKS:-all}"
mkdir -p "$OUT_ROOT"

# LoRA mode: merge adapter into base once, then eval the merged snapshot.
if [ -n "${ADAPTER:-}" ]; then
    MERGED="$OUT_ROOT/merged_${TAG}"
    if [ ! -f "$MERGED/config.json" ]; then
        echo "--- merging adapter $ADAPTER into base ---"
        python - "$ADAPTER" "$MERGED" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
adapter, out = sys.argv[1], sys.argv[2]
base = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged ->", out)
PY
    fi
    MODEL_PATH="$MERGED"
fi
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH=<checkpoint dir or HF id> (or ADAPTER=...)}"

echo "=========================================="
echo "Tulu3 suite eval | arm=$TAG"
echo "  model=$MODEL_PATH"
echo "  out=$OUTPUT_DIR  job=${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "=========================================="

python -c "import vllm, lm_eval; print('vllm', vllm.__version__, '| lm-eval', lm_eval.__version__)"

export HF_ALLOW_CODE_EVAL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY
export TRANSFORMERS_VERBOSITY=error

VLLM_ARG="--use_vllm"
[ "${FORCE_HF_BACKEND:-0}" = "1" ] && VLLM_ARG=""

PYTHONUNBUFFERED=1 python src/evaluation/run_all_benchmarks.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --benchmarks $BENCHMARKS \
    --batch_size auto \
    --trust_remote_code \
    --apply_chat_template \
    --verbose \
    $VLLM_ARG ${EVAL_EXTRA_ARGS:-}

echo "Finished: $(date)  results: $OUTPUT_DIR"

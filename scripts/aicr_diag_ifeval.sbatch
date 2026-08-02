#!/bin/bash
# One-off diagnostic: run evaluate_ifeval DIRECTLY (no output-capturing
# subprocess wrapper) with streaming stdout so the crash that kills the ifeval
# worker inside the suite is visible. limit=8 docs.
#SBATCH --job-name=diag_ifeval
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/diag_ifeval_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh
W=/work/neu/p2026_0038_neu/xie_yiyi
conda deactivate || true
conda activate "$W/envs/rl_casino_eval"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
CUDA_ROOT=/apps/aicr/packages/cuda/13.1.1/extrybz
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_MODULE_LOADING=LAZY

cd "$W/rc-sparse-speed"
PYTHONUNBUFFERED=1 python - <<'PY'
import sys
sys.path.insert(0, "src")
from evaluation.ifeval_evaluator import evaluate_ifeval
r = evaluate_ifeval(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    model="vllm", limit=8, batch_size="auto",
    trust_remote_code=True, apply_chat_template=True,
)
res = r.get("results", {}) if isinstance(r, dict) else r
print("IFEVAL_DIAG_RESULT:", res)
PY
echo "DIAG_IFEVAL_DONE"

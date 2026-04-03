#!/bin/bash
#SBATCH --job-name=grpo_masks_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/grpo_masks_gpu_%j.out
#SBATCH --error=logs/grpo_masks_gpu_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID:-N/A}"
echo "Node     : ${SLURMD_NODENAME:-N/A}"
echo "Started  : $(date)"
echo "========================================"

REPO_DIR="${SLURM_SUBMIT_DIR:-/home/xie.yiyi/Reinforcement-Casino}"
MASK_SCRIPT="$REPO_DIR/scripts/gen_grpo_masks_and_plot.sh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"

if [ ! -f "$MASK_SCRIPT" ]; then
    echo "[error] Missing script: $MASK_SCRIPT"
    exit 1
fi

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "[error] CONTAINER_IMAGE is not set."
    echo "        Re-submit with: --export=ALL,CONTAINER_IMAGE=/path/to/your_container.sif"
    exit 1
fi

if command -v apptainer >/dev/null 2>&1; then
    RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME="singularity"
else
    echo "[error] Neither apptainer nor singularity found on this node."
    exit 1
fi

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/masks/grpo_verify"

echo "[config] REPO_DIR=$REPO_DIR"
echo "[config] CONTAINER_IMAGE=$CONTAINER_IMAGE"
echo "[config] RUNTIME=$RUNTIME"
echo "[config] MODEL=${MODEL:-google/gemma-3-270m-it}"
echo "[config] N_SAMPLES=${N_SAMPLES:-64}"
echo "[config] SPARSITY=${SPARSITY:-90.0}"
echo "[config] SKIP_DPO=${SKIP_DPO:-1}"

INNER_CMD=$(cat <<EOF
set -euo pipefail
cd "$REPO_DIR"

python - <<'PY'
import ctypes
import torch

print(f"[preflight] torch={torch.__version__} cuda_build={torch.version.cuda}")
try:
    ctypes.CDLL("libcuda.so.1")
    print("[preflight] libcuda.so.1: OK")
except OSError as e:
    print(f"[preflight] libcuda.so.1: FAIL -> {e}")
    raise SystemExit(1)

ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f"[preflight] torch.cuda.is_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if not ok:
    raise SystemExit("[preflight] CUDA unavailable inside container runtime.")
print(f"[preflight] CUDA device 0: {torch.cuda.get_device_name(0)}")
PY

FORCE_GPU=1 \
MODEL="${MODEL:-google/gemma-3-270m-it}" \
N_SAMPLES="${N_SAMPLES:-64}" \
SPARSITY="${SPARSITY:-90.0}" \
MASK_DIR="${MASK_DIR:-masks/grpo_verify}" \
SKIP_DPO="${SKIP_DPO:-1}" \
DPO_SNIP_MASK="${DPO_SNIP_MASK:-masks/cold_start_snip_90pct.pt}" \
DPO_CAV_MASK="${DPO_CAV_MASK:-masks/cold_start_cav_90pct.pt}" \
DPO_FISHER_MASK="${DPO_FISHER_MASK:-masks/grpo_verify/fisher_dpo.pt}" \
CPU_THREADS="${CPU_THREADS:-}" \
bash "$MASK_SCRIPT"
EOF
)

"$RUNTIME" exec --nv \
    --bind "$REPO_DIR:$REPO_DIR" \
    "$CONTAINER_IMAGE" \
    /bin/bash -lc "$INNER_CMD"

echo "========================================"
echo "Finished : $(date)"
echo "========================================"

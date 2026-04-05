#!/bin/bash
# Stage 2a/5: warm masks only (delta-based magnitude / momentum / fisher).
# Uses a GPU so delta accumulation + mask thresholding run on-device (see choose_score_device in
# even_better_mask_finder.py). For CPU-only sites: sbatch with -p … and
# export RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu before submit.
#
# Northeastern Explorer: default partition `gpu` + H200 (match stage 1). Override: sbatch -p …
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --job-name=pipe_p2a_warm
#SBATCH --mem=128G
#SBATCH --output=logs/pipeline_%j_p2a_masks_warm.out
#SBATCH --error=logs/pipeline_%j_p2a_masks_warm.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

export PIPELINE_MASK_PHASE=warm
# Do not set RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu here — pipeline_setup defaults to cuda so
# warm mask scoring uses the GPU. (Previously this job used the CPU partition + forced cpu,
# which made CUDA unavailable and hid GPU utilization.)

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

# Fail fast if this job requested a GPU but PyTorch cannot see CUDA — idle-GPU policies
# often cancel those jobs; warm scoring defaults to RL_CASINO_WARM_MASK_SCORE_DEVICE=cuda.
# Skip when intentionally using CPU scoring (OOM / CPU partition): set RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu.
if [ "${RL_CASINO_WARM_MASK_SCORE_DEVICE:-cuda}" != "cpu" ] && [ "${PIPELINE_SKIP_WARM_GPU_PREFLIGHT:-0}" != "1" ]; then
  echo "=== GPU preflight (warm masks use CUDA for delta accumulation when available) ==="
  "$TRAIN_PY" -c "
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        'ERROR: CUDA not visible to PyTorch. Use a GPU Slurm allocation (--gres=gpu), '
        'submit from a GPU partition, and ensure this node exposes the device. '
        'To run warm scoring on CPU instead, export RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu before sbatch.'
    )
# Short matmul so nvidia-smi / schedulers see real GPU work (not just idle allocation).
a = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)
_ = (a @ a).sum()
torch.cuda.synchronize()
print('  OK:', torch.cuda.get_device_name(0), '| device capability in use')
del a
torch.cuda.empty_cache()
"
fi

echo "===== STAGE 2a/5: warm masks only (${RUN_ID}) ====="
run_masks

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: expected SLURM_JOB_ID for chaining" >&2
  exit 1
fi
jid=$(sbatch --parsable \
  --dependency=afterok:"${SLURM_JOB_ID}" \
  --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
  "${REPO_ROOT}/scripts/pipeline_stage_02b_masks_cold.sh")
echo "Chained next stage: pipeline_stage_02b_masks_cold.sh → Slurm job ${jid} (afterok:${SLURM_JOB_ID})"

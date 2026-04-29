#!/bin/bash
# Toy job: generate a block mask, persist it under OUT_BASE, and sanity-check active-block indexing.
# Submit from repo root:
#   sbatch scripts/toy_mask_sanity_bsr.sh
#
# Outputs under OUT_BASE (defaults to scratch):
# - mask_saved.pt
# - mask_sanity.txt
# - mask_block_profile.txt

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=toy_bsr_mask_sanity
#SBATCH --output=logs/toy_bsr_mask_sanity_%j.out
#SBATCH --error=logs/toy_bsr_mask_sanity_%j.err

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

TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export SPARSITY_PCT="${SPARSITY_PCT:-99.75}"
export MASK_SEED="${MASK_SEED:-425259}"
export BLOCK_SIZE_BSR="${BLOCK_SIZE_BSR:-16}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

echo "OUT_BASE=${OUT_BASE}"
echo "MODEL=${MODEL}"
echo "SPARSITY_PCT=${SPARSITY_PCT}  BLOCK_SIZE_BSR=${BLOCK_SIZE_BSR}  MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}  MASK_SEED=${MASK_SEED}"

"$TRAIN_PY" - <<'PY' | tee "${OUT_BASE}/mask_block_profile.txt"
import os
import torch
from src.utils.slurm_safe_log import slurm_safe_print
from src.full_training.h200_sparse_dpo_bsr_benchmark import generate_block_random_masks_cpu
from src.utils.mask_utils import save_masks
from src.utils.block_profiler import print_block_sparsity_profile

model = os.environ["MODEL"]
sparsity = float(os.environ["SPARSITY_PCT"])
seed = int(os.environ["MASK_SEED"])
block = int(os.environ["BLOCK_SIZE_BSR"])
min_keep = float(os.environ["MIN_LAYER_KEEP_RATIO"])
out_base = os.environ["OUT_BASE"]

masks = generate_block_random_masks_cpu(
    model_name=model,
    checkpoint_path=None,
    sparsity_percent=sparsity,
    seed=seed,
    mlp_only=False,
    min_layer_keep_ratio=min_keep,
    block_size=block,
)
bool_masks = {k: v.bool() if v.dtype != torch.bool else v for k, v in masks.items()}
print_block_sparsity_profile(bool_masks, block_size=block)

dst = os.path.join(out_base, "mask_saved.pt")
meta = {
    "method": "toy_mask_sanity_bsr",
    "sparsity_percent": sparsity,
    "seed": seed,
    "format": "torch_bool_binary",
    "block_size_bsr": block,
    "min_layer_keep_ratio": min_keep,
}
save_masks(bool_masks, dst, metadata=meta)
slurm_safe_print(f"✓ Saved mask to: {dst}")
PY

"$TRAIN_PY" - <<'PY' | tee "${OUT_BASE}/mask_sanity.txt"
import os
import torch
from src.utils.mask_manager import SparseMaskManager

mask_path = os.path.join(os.environ["OUT_BASE"], "mask_saved.pt")
mm = SparseMaskManager(mask_path, device=torch.device("cpu"))

pick = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "lm_head.weight",
]

print(f"mask_path={mask_path}")
for name in pick:
    if not mm.has_mask(name):
        print(f"SKIP {name}: no mask")
        continue
    m = mm.get_mask(name)
    M, N = int(m.shape[0]), int(m.shape[1])
    num_blocks_n = (N + 15) // 16
    ab = mm.get_active_block_indices(name)
    print(f"\n{name}")
    print(f"  shape={tuple(m.shape)} active_blocks={int(ab.numel())} dtype={ab.dtype} contiguous={ab.is_contiguous()}")
    if ab.numel() == 0:
        continue
    for i in [0, 1, 2, 3, 4]:
        b = int(ab[i].item())
        bm = b // num_blocks_n
        bn = b % num_blocks_n
        ok = (0 <= bm < (M + 15) // 16) and (0 <= bn < num_blocks_n)
        print(f"  ab[{i}]={b} -> (bm={bm}, bn={bn}) ok={ok}")
PY

echo "DONE. OUT_BASE=${OUT_BASE}"

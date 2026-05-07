#!/usr/bin/env bash
# Submit **independent** CPU→GPU chains: one Short job per sparsity (mask gen) → one Multigpu H200
# microbench each, linked only by per-chain `afterok`. Does **not** submit one giant prefetch that
# blocks every GPU.
#
# ``MASK_CHAIN_ROOT`` (maps to ``H200_BSR_MASK_CACHE``) is **only** a directory you choose on scratch so
# the driver can write ``masks/*.pt``. Default below is fine; rename to anything under ``/scratch/$USER``.
# Independent chains do not require a shared path — one root per sweep is just convenient.
#
# Usage (login node, repo root):
#   export SCRATCH_USER_ROOT="/scratch/${USER}"
#   export MASK_CHAIN_ROOT="/scratch/${USER}/rl_casino_h200_bsr/mask_chains_${USER}_$(date +%Y%m%d_%H%M%S)"
#   export CHAIN_SPARSITIES="${CHAIN_SPARSITIES:-50,75,90,99.75}"
#   ./scripts/submit_independent_mask_gpu_chains.sh
#
# Optional:
#   CHAIN_MASK_TYPES (default element,block)
#   STEPS TRIM_FRAC LR BLOCK_SIZE         (microbench timing settings)
#   MAX_TOTAL_NUMEL MAX_TENSORS SELECTION_ORDER CAP_BEHAVIOR
#       Tensor subset policy. Defaults reproduce the 97.5% paper row:
#         MAX_TOTAL_NUMEL=525000000  MAX_TENSORS=64  SELECTION_ORDER=model_order  CAP_BEHAVIOR=break
#   CHAIN_SKIP_CPU_IF_EXISTS=1            Reuse existing masks already in MASK_CHAIN_ROOT/masks/.

set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
MASK_CHAIN_ROOT="${MASK_CHAIN_ROOT:-${SCRATCH_USER_ROOT}/rl_casino_h200_bsr/mask_chains_${USER}_$(date +%Y%m%d_%H%M%S)}"
export H200_BSR_MASK_CACHE="${MASK_CHAIN_ROOT}"
mkdir -p "${H200_BSR_MASK_CACHE}/masks"

CHAIN_SPARSITIES="${CHAIN_SPARSITIES:-50,75,90,99.75}"
CHAIN_MASK_TYPES="${CHAIN_MASK_TYPES:-element,block}"

# Tensor subset selection — defaults match the historical 97.5% row policy.
MAX_TOTAL_NUMEL="${MAX_TOTAL_NUMEL:-525000000}"
MAX_TENSORS="${MAX_TENSORS:-64}"
SELECTION_ORDER="${SELECTION_ORDER:-model_order}"
CAP_BEHAVIOR="${CAP_BEHAVIOR:-break}"
CHAIN_SKIP_CPU_IF_EXISTS="${CHAIN_SKIP_CPU_IF_EXISTS:-0}"

IFS=',' read -ra _SP_LIST <<< "${CHAIN_SPARSITIES}"
echo "MASK_CHAIN_ROOT=${MASK_CHAIN_ROOT}"
echo "CHAIN_SPARSITIES=${CHAIN_SPARSITIES}"
echo "CHAIN_MASK_TYPES=${CHAIN_MASK_TYPES}"
echo "MAX_TOTAL_NUMEL=${MAX_TOTAL_NUMEL} MAX_TENSORS=${MAX_TENSORS} SELECTION_ORDER=${SELECTION_ORDER} CAP_BEHAVIOR=${CAP_BEHAVIOR}"
echo "CHAIN_SKIP_CPU_IF_EXISTS=${CHAIN_SKIP_CPU_IF_EXISTS}"

for raw in "${_SP_LIST[@]}"; do
  sp="${raw//[[:space:]]/}"
  [[ -z "${sp}" ]] && continue
  jid=$(sbatch --parsable \
    --export=ALL,CHAIN_SPARSITY="${sp}",H200_BSR_MASK_CACHE="${H200_BSR_MASK_CACHE}",CHAIN_MASK_TYPES="${CHAIN_MASK_TYPES}",MAX_TOTAL_NUMEL="${MAX_TOTAL_NUMEL}",MAX_TENSORS="${MAX_TENSORS}",SELECTION_ORDER="${SELECTION_ORDER}",CAP_BEHAVIOR="${CAP_BEHAVIOR}",CHAIN_SKIP_CPU_IF_EXISTS="${CHAIN_SKIP_CPU_IF_EXISTS}" \
    "${REPO_ROOT}/scripts/h200_bsr_mask_chain_cpu.slurm")
  echo "Submitted chain sp=${sp}  CPU_JOB=${jid}"
done

echo "Done. Each CPU job submits its own GPU microbench when masks finish. Outputs: logs/h200_bsr_mask_chain_cpu_*.out"

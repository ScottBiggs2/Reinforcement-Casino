#!/bin/bash
# Fire-and-forget chain for Olmo-3-7B delta-log ORACLE masks + sparse DPO.
#
# Submits (from the LOGIN node — these are just sbatch calls, no heavy work):
#   1. gen_oracle_deltalog_olmo3_7b_lr1.slurm  (build delta-log oracle masks at each sparsity)
#   2. one sparse_dpo_olmo3_7b_masked_lr1_500.slurm per sparsity, each --dependency=afterok:(1)
#
# Usage (run on the Explorer login node, from repo root):
#   bash scripts/chain_oracle_deltalog_olmo3_7b.sh <DENSE_RUN_ID>
#   # or override the sparsities:
#   SPARSITY_LIST="90 99" bash scripts/chain_oracle_deltalog_olmo3_7b.sh <DENSE_RUN_ID>
#
# DENSE_RUN_ID may be the full RUN_ID (e.g. 20260728_184922_8811836) or the bare
# Slurm jobid (e.g. 8811836) — the mask-gen script resolves it either way.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DENSE_RUN_ID="${1:-${DENSE_RUN_ID:-}}"
SPARSITY_LIST="${SPARSITY_LIST:-90 99}"
ORACLE_TARGET_STEP="${ORACLE_TARGET_STEP:-500}"

if [ -z "${DENSE_RUN_ID}" ]; then
  echo "usage: bash scripts/chain_oracle_deltalog_olmo3_7b.sh <DENSE_RUN_ID>" >&2
  echo "       (or set DENSE_RUN_ID / SPARSITY_LIST in the environment)" >&2
  exit 1
fi

# Mask output location (must match gen_oracle_deltalog_olmo3_7b_lr1.slurm).
SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
MASK_OUT_BASE="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
SUBDIR="allenai_olmo_3_7b_instruct_light_r1"
MASK_DIR="${MASK_OUT_BASE}/olmo3_7b_light_r1"

# sbatch with retry — survives transient slurmctld "Unexpected message received".
sbatch_retry() {
  local out n=0
  while true; do
    if out=$(sbatch --parsable "$@" 2>&1); then
      printf '%s\n' "$out"
      return 0
    fi
    n=$((n + 1))
    if [ "$n" -ge 5 ]; then
      echo "ERROR: sbatch failed after ${n} attempts: ${out}" >&2
      return 1
    fi
    echo "  (sbatch retry ${n} after transient error: ${out})" >&2
    sleep 15
  done
}

echo "Chaining Olmo-3-7B delta-log oracle: DENSE_RUN_ID=${DENSE_RUN_ID}  sparsities='${SPARSITY_LIST}'"

# 1. Mask generation
MASKJOB=$(sbatch_retry \
  --export=ALL,DENSE_RUN_ID="${DENSE_RUN_ID}",SPARSITY_LIST="${SPARSITY_LIST}",ORACLE_TARGET_STEP="${ORACLE_TARGET_STEP}" \
  scripts/gen_oracle_deltalog_olmo3_7b_lr1.slurm)
echo "  [mask-gen]  job ${MASKJOB}"

# 2. One sparse DPO run per sparsity, gated on afterok of the mask job.
#    Mask-gen is failure-tolerant per sparsity; each sparse job independently checks
#    its own MASK_FILE exists, so a single degenerate sparsity won't block the others.
for SP in ${SPARSITY_LIST}; do
  MASK_FILE="${MASK_DIR}/oracle_deltalog_${SUBDIR}_sp${SP}pct_step${ORACLE_TARGET_STEP}.pt"
  SJOB=$(sbatch_retry \
    --dependency=afterok:"${MASKJOB}" \
    --export=ALL,MASK_FILE="${MASK_FILE}",MASK_LABEL="oracle_deltalog_sp${SP}" \
    scripts/sparse_dpo_olmo3_7b_masked_lr1_500.slurm)
  echo "  [sparse sp${SP}]  job ${SJOB}  (afterok:${MASKJOB})"
done

echo ""
echo "Submitted. Chain: mask-gen (${MASKJOB}) → sparse runs (afterok). Safe to log off."
echo "Check in the morning:  squeue -u \$USER   and   logs/mk_oracle_dl_olmo3_*.out"

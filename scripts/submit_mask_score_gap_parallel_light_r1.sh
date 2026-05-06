#!/usr/bin/env bash
# One-shot submit: cache → baseline_shard → milestone array → merge+plots.
# Run from the cluster login node (not inside a batch job), after exporting paths/tokens like the non-parallel script.
#
# Required: cd to repo root (contains scripts/ and src/). Optional: export OUT_DIR, MAGNITUDE_MILESTONES, CERT_*, etc.
#
#   export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_parallel/run1
#   bash scripts/submit_mask_score_gap_parallel_light_r1.sh
#
# Skip cache stage if magnitude_caches/ already exist:
#   SKIP_CACHE_STAGE=1 bash scripts/submit_mask_score_gap_parallel_light_r1.sh
#
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"

SBATCH_SCRIPT="${REPO_ROOT}/scripts/slurm_mask_score_gap_parallel_light_r1.slurm"
if [ ! -f "${SBATCH_SCRIPT}" ]; then
  echo "ERROR: missing ${SBATCH_SCRIPT}" >&2
  exit 1
fi

MAGNITUDE_MILESTONES="${MAGNITUDE_MILESTONES:-50,100,150,200}"
IFS=',' read -r -a _MS <<< "${MAGNITUDE_MILESTONES}"
N_MILE="${#_MS[@]}"
if [ "${N_MILE}" -lt 1 ]; then
  echo "ERROR: empty MAGNITUDE_MILESTONES" >&2
  exit 1
fi
LAST_IDX=$((N_MILE - 1))

EXP_COMMON="ALL"
# Caller should already export OUT_DIR, HF_TOKEN, CKPT500_DIR, DELTA_LOG_DIR, CERT_*, etc.; --export=ALL forwards them.

if [ "${SKIP_CACHE_STAGE:-0}" = "1" ]; then
  echo "SKIP_CACHE_STAGE=1: submitting baseline_shard without cache job"
  J_BASE=$(sbatch --parsable --export="${EXP_COMMON}",MASK_GAP_STAGE=baseline_shard "${SBATCH_SCRIPT}")
  echo "baseline_shard job_id=${J_BASE}"
else
  J_CACHE=$(sbatch --parsable --export="${EXP_COMMON}",MASK_GAP_STAGE=cache_only "${SBATCH_SCRIPT}")
  echo "cache_only job_id=${J_CACHE}"
  J_BASE=$(sbatch --parsable --dependency=afterok:"${J_CACHE}" --export="${EXP_COMMON}",MASK_GAP_STAGE=baseline_shard "${SBATCH_SCRIPT}")
  echo "baseline_shard job_id=${J_BASE} (after cache ${J_CACHE})"
fi

J_MILE=$(sbatch --parsable --dependency=afterok:"${J_BASE}" --array="0-${LAST_IDX}" --export="${EXP_COMMON}",MASK_GAP_STAGE=milestone_shard "${SBATCH_SCRIPT}")
echo "milestone_shard array job_id=${J_MILE} (tasks 0-${LAST_IDX})"

J_MERGE=$(sbatch --parsable --dependency=afterok:"${J_MILE}" --export="${EXP_COMMON}",MASK_GAP_STAGE=merge_shards "${SBATCH_SCRIPT}")
echo "merge_shards job_id=${J_MERGE} (after milestone array ${J_MILE})"
echo "Final artifacts under OUT_DIR after job ${J_MERGE} completes."

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
# --- Scheduler tuning (optional env overrides) ---
# This script passes different sbatch resource flags per stage so merge/plots do not request 256G/8h.
# Defaults below are aggressive-but-usual for Llama-8B bf16 + CERT_GLOBAL_MODE=stream; adjust after measuring:
#   sacct -j JOBID --format=JobID,JobName,MaxRSS,Elapsed,State -P
# Override any stage, e.g.: export MGAP_SBATCH_MERGE="--mem=64G --time=02:00:00 --cpus-per-task=4"
#
# MGAP_SBATCH_EXTRA="--partition=short"   # optional: appended to every sbatch line
# MGAP_SBATCH_CACHE="--mem=... --time=..."       # cache_only
# MGAP_SBATCH_BASE="--mem=... --time=..."        # baseline_shard
# MGAP_SBATCH_MILESTONE="--mem=... --time=..."   # milestone_shard array
# MGAP_SBATCH_MERGE="--mem=... --time=..."       # merge_shards + plots (no checkpoint load — cheap)
#
set -euo pipefail

MGAP_SBATCH_EXTRA="${MGAP_SBATCH_EXTRA:-}"
# Heavy stages: two full state dicts + streaming cert (defaults trimmed from 256G — raise if OOM)
MGAP_SBATCH_CACHE="${MGAP_SBATCH_CACHE:---mem=192G --time=06:00:00 --cpus-per-task=8}"
MGAP_SBATCH_BASE="${MGAP_SBATCH_BASE:---mem=192G --time=08:00:00 --cpus-per-task=8}"
MGAP_SBATCH_MILESTONE="${MGAP_SBATCH_MILESTONE:---mem=192G --time=08:00:00 --cpus-per-task=8}"
# merge_parallel_shards + matplotlib: small RSS vs loading checkpoints
MGAP_SBATCH_MERGE="${MGAP_SBATCH_MERGE:---mem=48G --time=01:00:00 --cpus-per-task=2}"

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
  # shellcheck disable=SC2086
  J_BASE=$(sbatch --parsable ${MGAP_SBATCH_EXTRA} ${MGAP_SBATCH_BASE} --export="${EXP_COMMON}",MASK_GAP_STAGE=baseline_shard "${SBATCH_SCRIPT}")
  echo "baseline_shard job_id=${J_BASE}"
else
  # shellcheck disable=SC2086
  J_CACHE=$(sbatch --parsable ${MGAP_SBATCH_EXTRA} ${MGAP_SBATCH_CACHE} --export="${EXP_COMMON}",MASK_GAP_STAGE=cache_only "${SBATCH_SCRIPT}")
  echo "cache_only job_id=${J_CACHE}"
  # shellcheck disable=SC2086
  J_BASE=$(sbatch --parsable ${MGAP_SBATCH_EXTRA} ${MGAP_SBATCH_BASE} --dependency=afterok:"${J_CACHE}" --export="${EXP_COMMON}",MASK_GAP_STAGE=baseline_shard "${SBATCH_SCRIPT}")
  echo "baseline_shard job_id=${J_BASE} (after cache ${J_CACHE})"
fi

# shellcheck disable=SC2086
J_MILE=$(sbatch --parsable ${MGAP_SBATCH_EXTRA} ${MGAP_SBATCH_MILESTONE} --dependency=afterok:"${J_BASE}" --array="0-${LAST_IDX}" --export="${EXP_COMMON}",MASK_GAP_STAGE=milestone_shard "${SBATCH_SCRIPT}")
echo "milestone_shard array job_id=${J_MILE} (tasks 0-${LAST_IDX})"

# shellcheck disable=SC2086
J_MERGE=$(sbatch --parsable ${MGAP_SBATCH_EXTRA} ${MGAP_SBATCH_MERGE} --dependency=afterok:"${J_MILE}" --export="${EXP_COMMON}",MASK_GAP_STAGE=merge_shards "${SBATCH_SCRIPT}")
echo "merge_shards job_id=${J_MERGE} (after milestone array ${J_MILE})"
echo "Final artifacts under OUT_DIR after job ${J_MERGE} completes."

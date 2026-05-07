#!/usr/bin/env bash
# Mask score-gap / parallel DAG status (login node).
#
# Usage:
#   OUT_DIR=/scratch/$USER/.../run2_parallel bash scripts/check_mask_score_gap_status.sh
#   JOBID=6604704 OUT_DIR=/scratch/$USER/.../run2_parallel bash scripts/check_mask_score_gap_status.sh
#
# Prints: magnitude caches, parallel shard progress, final artifacts (monolithic vs parallel-complete),
# optional sacct + log phase hints for JOBID (baseline Slurm id).

set -euo pipefail

if [ -z "${OUT_DIR:-}" ]; then
  echo "ERROR: set OUT_DIR=/path/to/analysis/run" >&2
  exit 1
fi

echo "=== OUT_DIR=${OUT_DIR} ==="
if [ ! -d "${OUT_DIR}" ]; then
  echo "WARN: OUT_DIR does not exist yet."
  exit 0
fi

echo ""
echo "--- magnitude_caches (delta stream skip when all OK) ---"
MC="${OUT_DIR}/magnitude_caches"
_MS="${MAGNITUDE_MILESTONES:-50,100,150,200}"
if [ -d "${MC}" ]; then
  ls -lah "${MC}" 2>/dev/null || true
  IFS=',' read -r -a _steps <<< "${_MS}"
  for s in "${_steps[@]}"; do
    s="${s// /}"
    [ -z "${s}" ] && continue
    f="${MC}/mag_aggregate_step_${s}.pt"
    [ -f "$f" ] && echo "OK   step ${s}" || echo "MISS step ${s}"
  done
else
  echo "(no magnitude_caches/)"
fi

echo ""
echo "--- parallel_shards (baseline must finish before milestone array runs) ---"
PSD="${OUT_DIR}/parallel_shards"
if [ -d "${PSD}" ]; then
  ls -lah "${PSD}" 2>/dev/null || true
  # merge_parallel_shards expects baseline_shard.pt + baseline_shard.pt.done (see mask_score_gap_analysis.py)
  for f in baseline_shard.pt baseline_shard.pt.done; do
    if [ -f "${PSD}/${f}" ]; then
      echo "OK   ${PSD}/${f}"
    else
      echo "MISS ${PSD}/${f}"
    fi
  done
  shopt -s nullglob
  for f in "${PSD}"/milestone_*_shard.pt "${PSD}"/milestone_*_shard.pt.done; do
    [ -e "$f" ] && echo "OK   $(basename "$f")"
  done
  shopt -u nullglob
else
  echo "(no parallel_shards/)"
fi

echo ""
echo "--- final artifacts (monolithic OR merge_shards complete) ---"
for name in mask_score_gap_summary.csv mask_score_gap_by_layer.csv mask_score_gap_gap_diagnostics.json mask_score_gap_histograms.npz; do
  p="${OUT_DIR}/${name}"
  if [ -f "$p" ]; then
    echo "OK   ${name} ($(du -h "$p" | cut -f1))"
  else
    echo "MISS ${name}"
  fi
done
if [ -d "${OUT_DIR}/figures" ]; then
  nfig=$(find "${OUT_DIR}/figures" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "figures/: ${nfig} files (top-level)"
else
  echo "MISS figures/"
fi

echo ""
echo "--- run metadata ---"
if [ -f "${OUT_DIR}/mask_score_gap_run.json" ]; then
  head -c 800 "${OUT_DIR}/mask_score_gap_run.json"
  echo ""
else
  echo "(no mask_score_gap_run.json)"
fi

if [ -n "${JOBID:-}" ]; then
  echo ""
  echo "--- sacct JOBID=${JOBID} ---"
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "${JOBID}" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem -P 2>/dev/null || echo "(sacct failed)"
  else
    echo "(sacct not in PATH)"
  fi

  echo ""
  echo "--- log phase hints (repo logs/) ---"
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  for suffix in out err; do
    log="${REPO_ROOT}/logs/mask_score_gap_parallel_${JOBID}.${suffix}"
    if [ -f "${log}" ]; then
      echo ">>> ${log} (last 30 lines)"
      tail -n 30 "${log}"
      echo ""
      echo "Phase grep (${suffix}):"
      grep -E 'Computing random-mask|Cert/margins:|processed key|Milestone @|execution_mode|Done stage|Traceback|ERROR|Killed' "${log}" 2>/dev/null | tail -n 15 || echo "(no matches)"
    fi
  done
  if [ ! -f "${REPO_ROOT}/logs/mask_score_gap_parallel_${JOBID}.out" ]; then
    echo "No ${REPO_ROOT}/logs/mask_score_gap_parallel_${JOBID}.out — try logs under \$PWD if job submitted elsewhere."
  fi
fi

echo ""
echo "Tips:"
echo "  tail -f logs/mask_score_gap_parallel_<JOBID>.out"
echo "  Parallel array+merge stay PD (Dependency) until baseline_shard completes."
echo "  If summary.csv exists, you may skip redundant parallel merge / duplicate monolithic work."

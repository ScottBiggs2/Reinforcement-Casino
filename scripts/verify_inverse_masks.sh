#!/usr/bin/env bash
# Verify complement masks (*_inverse.pt) for a pipeline RUN_ID.
# Paths match scripts/pipeline_common.sh defaults.
#
# Usage (from repo root):
#   bash scripts/verify_inverse_masks.sh <PIPELINE_RUN_ID>
#
# Optional env: SCRATCH_USER_ROOT, MASK_OUT_BASE, REPO_ROOT

set -euo pipefail

RUN_ID="${1:?usage: $0 <PIPELINE_RUN_ID>}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
MASK_OUT_BASE="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
mask_dir="${MASK_OUT_BASE}/${RUN_ID}"
log_file="${REPO_ROOT}/logs/full_pipeline_masks_${RUN_ID}.log"

echo "=== Inverse mask check for RUN_ID=${RUN_ID} ==="
echo "Mask directory: ${mask_dir}"

if [ ! -d "$mask_dir" ]; then
  echo "ERROR: mask directory does not exist." >&2
  exit 1
fi

shopt -s nullglob
n_inv=0
n_primary=0
for f in "${mask_dir}"/*.pt; do
  base=$(basename "$f")
  if [[ "$base" == *_inverse.pt ]]; then
    n_inv=$((n_inv + 1))
  else
    n_primary=$((n_primary + 1))
  fi
done
shopt -u nullglob

echo "Non-inverse *.pt (primaries): ${n_primary}"
echo "*_inverse.pt files:            ${n_inv}"

if [ "$n_primary" -eq 0 ]; then
  echo "WARNING: no primary mask .pt files found." >&2
else
  if [ "$n_primary" -eq "$n_inv" ]; then
    echo "OK: inverse count matches primary count (one complement per primary expected)."
  else
    echo "NOTE: primary count (${n_primary}) != inverse count (${n_inv}). Check mask phase logs for failed complement steps."
  fi
fi

echo ""
echo "Sample *_inverse.pt:"
ls -1 "${mask_dir}"/*_inverse.pt 2>/dev/null | head -n 20 || echo "(none)"

echo ""
if [ -f "$log_file" ]; then
  echo "Log: ${log_file}"
  echo "--- complement / inverse lines (grep) ---"
  grep -E "Complement masks|complement \(inverse\)|OK: complement|SKIP complement|invert_mask" "$log_file" 2>/dev/null || echo "(no matching lines)"
else
  echo "Mask log not found: ${log_file} (run may be on another host or log name differs)"
fi

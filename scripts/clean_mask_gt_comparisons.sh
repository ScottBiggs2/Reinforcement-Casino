#!/bin/bash
# Remove scratch outputs from mask analysis vs ground truth (JSON/CSV/PNGs under
# comparisons_vs_ground_truth). Does NOT delete .pt mask files in MASK_ANALYSIS_DIR.
#
# Usage:
#   bash scripts/clean_mask_gt_comparisons.sh /scratch/.../tulu3_500_h200_fresh_0409_again
set -euo pipefail
MASK_ANALYSIS_DIR="${1:?Usage: $0 MASK_ANALYSIS_DIR}"
COMP="${MASK_ANALYSIS_DIR}/comparisons_vs_ground_truth"
rm -rf "${COMP}"
mkdir -p "${COMP}/plots"
echo "Removed and recreated: ${COMP}"

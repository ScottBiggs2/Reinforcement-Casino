#!/bin/bash
# ============================================================================
# Llama-3.1-8B CKA Analysis: Cold masks vs Ground Truth
# ============================================================================
# Ground truth = warm magnitude mask (cumulative |delta| from step 0 to final).
# It knows exactly which weights changed and by how much — total oracle/cheating.
#
# Question: How close do cold CAV/SNIP (zero training info) get to the ground
# truth subnetwork's internal representations? And why do they beat cold Fisher?
#
# Comparisons (all mask_vs_mask — same calibration data, layer-wise CKA):
#
#   Group A — Cold vs Ground Truth (core question):
#     cold_snip  vs warm_magnitude  (DPO & GRPO)
#     cold_cav   vs warm_magnitude  (DPO & GRPO)
#     cold_fisher vs warm_magnitude (DPO & GRPO)  ← baseline: Fisher loses
#     → If CAV/SNIP have higher CKA to ground truth than Fisher does,
#       they are better at predicting which circuits matter for training.
#
#   Group B — Representational Fidelity (original vs masked):
#     original vs cold_{snip,cav,fisher}  (DPO)
#     original vs warm_magnitude          (DPO)
#     → Shows how much each method distorts the pretrained model.
#       Ground truth should distort most (it targets change); if CAV/SNIP
#       distort similarly to ground truth, they're finding the same circuits.
#
#   Group C — Cross-method (CAV/SNIP similarity):
#     cold_snip vs cold_cav  (DPO & GRPO)
#     → Are the two best cold methods finding the same subnetwork?
#
# Masks:
#   Cold:         /scratch/$USER/rl_casino_masks/llama8b_cold/
#   Ground truth: /scratch/$USER/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt  (DPO)
#                 /scratch/$USER/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt (GRPO)
#
# Output: /scratch/$USER/rl_casino_outputs/cka_results_llama8b/
#
# Usage:
#   bash scripts/run_llama8b_cka_analysis.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# ── Configuration ──────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"
N_SAMPLES="${N_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"

SCRATCH_ROOT="/scratch/${USER}"

# Mask locations
COLD="${SCRATCH_ROOT}/rl_casino_masks/llama8b_cold"

# Ground truth masks (warm magnitude — the oracle)
GT_DPO="${SCRATCH_ROOT}/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt"
GT_GRPO="${SCRATCH_ROOT}/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt"

# Output
CKA_OUTPUT_DIR="${SCRATCH_ROOT}/rl_casino_outputs/cka_results_llama8b"

echo "=============================================="
echo " Llama-8B CKA: Cold masks vs Ground Truth"
echo "=============================================="
echo "MODEL:         $MODEL"
echo "COLD MASKS:    $COLD"
echo "GT (DPO):      $GT_DPO"
echo "GT (GRPO):     $GT_GRPO"
echo "OUTPUT:        $CKA_OUTPUT_DIR"
echo "=============================================="

# ── Build SLURM job script ────────────────────────────────────────────────

JOB_SCRIPT="${REPO_ROOT}/logs/job_cka_llama8b.sh"
cat > "$JOB_SCRIPT" << 'OUTER_EOF'
#!/bin/bash
#SBATCH --job-name=llama8b_cka
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/llama8b_cka_%j.out
#SBATCH --error=logs/llama8b_cka_%j.err

set -euo pipefail

source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/$USER/.conda/envs/rl_casino \
  || conda activate /scratch/$USER/conda_envs/rl_casino

export HF_HOME="/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/$USER/hf_cache/datasets"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

cd "${SLURM_SUBMIT_DIR}"

OUTER_EOF

# Append variable-interpolated portion
cat >> "$JOB_SCRIPT" << EOF

MODEL="${MODEL}"
N_SAMPLES=${N_SAMPLES}
BATCH_SIZE=${BATCH_SIZE}
MAX_LENGTH=${MAX_LENGTH}

COLD="${COLD}"
GT_DPO="${GT_DPO}"
GT_GRPO="${GT_GRPO}"
CKA_OUTPUT_DIR="${CKA_OUTPUT_DIR}"

mkdir -p "\$CKA_OUTPUT_DIR"

PASSED=0
FAILED=0
SKIPPED=0

run_cka() {
  local TAG="\$1" MASK_A="\$2" MASK_B="\$3" COMPARE="\$4"
  local OUT_JSON="\${CKA_OUTPUT_DIR}/cka_\${TAG}.json"

  if [ -f "\$OUT_JSON" ]; then
    echo "SKIP \$TAG: already exists"
    ((SKIPPED++)) || true
    return 0
  fi
  if [ ! -f "\$MASK_A" ]; then
    echo "SKIP \$TAG: mask_a missing: \$MASK_A"
    ((SKIPPED++)) || true
    return 0
  fi
  if [ ! -f "\$MASK_B" ]; then
    echo "SKIP \$TAG: mask_b missing: \$MASK_B"
    ((SKIPPED++)) || true
    return 0
  fi

  echo ""
  echo "=== CKA: \$TAG (mode: \$COMPARE) ==="
  echo "  A: \$MASK_A"
  echo "  B: \$MASK_B"

  if python src/cold_start/mask_to_cka.py "\$MASK_A" "\$MASK_B" \\
      --model_name "\$MODEL" \\
      --compare "\$COMPARE" \\
      --device cuda \\
      --n_samples "\$N_SAMPLES" \\
      --batch_size "\$BATCH_SIZE" \\
      --max_length "\$MAX_LENGTH" \\
      --seed 42 \\
      -o "\$OUT_JSON"; then
    echo "OK: \$TAG"
    ((PASSED++)) || true
  else
    echo "FAILED: \$TAG"
    ((FAILED++)) || true
  fi
}

# ══════════════════════════════════════════════════════════════════════════
# Group A: Cold vs Ground Truth (the core question)
#   Higher CKA to ground truth = better at predicting training-relevant circuits
# ══════════════════════════════════════════════════════════════════════════
echo "=============================================="
echo " Group A: Cold methods vs Ground Truth"
echo " (warm magnitude = oracle that saw all deltas)"
echo "=============================================="

# DPO ground truth comparisons
for METHOD in snip cav fisher; do
  run_cka "vs_gt_cold_\${METHOD}_dpo" \\
    "\${COLD}/cold_\${METHOD}_dpo.pt" \\
    "\$GT_DPO" \\
    "mask_vs_mask"
done

# GRPO ground truth comparisons
for METHOD in snip cav fisher; do
  run_cka "vs_gt_cold_\${METHOD}_grpo" \\
    "\${COLD}/cold_\${METHOD}_grpo.pt" \\
    "\$GT_GRPO" \\
    "mask_vs_mask"
done

# ══════════════════════════════════════════════════════════════════════════
# Group B: Representational Fidelity (original model vs masked)
#   How much does each method distort the pretrained representations?
#   If CAV/SNIP distort similarly to ground truth → same circuits targeted
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo " Group B: Representational Fidelity"
echo " (original model vs each masked subnetwork)"
echo "=============================================="

# Cold methods (DPO)
for METHOD in snip cav fisher; do
  run_cka "fidelity_cold_\${METHOD}_dpo" \\
    "\${COLD}/cold_\${METHOD}_dpo.pt" \\
    "\${COLD}/cold_\${METHOD}_dpo.pt" \\
    "original_vs_a"
done

# Ground truth (DPO) — the reference distortion level
run_cka "fidelity_gt_dpo" \\
  "\$GT_DPO" \\
  "\$GT_DPO" \\
  "original_vs_a"

# ══════════════════════════════════════════════════════════════════════════
# Group C: Cross-method (are SNIP and CAV finding the same subnetwork?)
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo " Group C: SNIP vs CAV similarity"
echo " (are the two best cold methods converging?)"
echo "=============================================="

for MODE in dpo grpo; do
  run_cka "cross_snip_vs_cav_\${MODE}" \\
    "\${COLD}/cold_snip_\${MODE}.pt" \\
    "\${COLD}/cold_cav_\${MODE}.pt" \\
    "mask_vs_mask"
done

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo " CKA Analysis Complete!"
echo "=============================================="
echo " Passed:  \$PASSED"
echo " Failed:  \$FAILED"
echo " Skipped: \$SKIPPED"
echo ""
echo " Results: \$CKA_OUTPUT_DIR/"
ls -la "\$CKA_OUTPUT_DIR"/cka_*.json 2>/dev/null || echo "(no results yet)"
echo ""
echo " Key comparisons to check:"
echo "   vs_gt_cold_snip_dpo   — SNIP → GT alignment (DPO)"
echo "   vs_gt_cold_cav_dpo    — CAV  → GT alignment (DPO)"
echo "   vs_gt_cold_fisher_dpo — Fisher → GT alignment (DPO, expect lowest)"
echo "=============================================="

# ── Auto-generate summary plot ────────────────────────────────────────────
echo ""
echo "Generating summary visualization..."
python scripts/plot_cka_vs_ground_truth.py \\
  --results_dir "\$CKA_OUTPUT_DIR" \\
  -o "\${CKA_OUTPUT_DIR}/cka_summary.png" \\
  || echo "WARNING: plot generation failed (non-fatal)"
EOF

chmod +x "$JOB_SCRIPT"

echo ""
echo "Submitting CKA analysis job..."
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
echo ""
echo "=============================================="
echo " SUBMITTED: SLURM job $JOB_ID"
echo "=============================================="
echo ""
echo "Comparisons (12 total):"
echo ""
echo "  Group A — Cold vs Ground Truth (6 runs):"
echo "    cold_{snip,cav,fisher} vs warm_magnitude  x {DPO, GRPO}"
echo "    → expect: snip/cav CKA > fisher CKA to ground truth"
echo ""
echo "  Group B — Representational Fidelity (4 runs):"
echo "    original vs cold_{snip,cav,fisher}_dpo"
echo "    original vs warm_magnitude_dpo"
echo "    → compare distortion profiles"
echo ""
echo "  Group C — SNIP vs CAV (2 runs):"
echo "    cold_snip vs cold_cav  x {DPO, GRPO}"
echo "    → do the best methods agree on which circuits matter?"
echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    tail -f logs/llama8b_cka_\${JOB_ID}.out"
echo "=============================================="

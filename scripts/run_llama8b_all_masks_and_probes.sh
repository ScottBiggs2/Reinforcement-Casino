#!/bin/bash
# ============================================================================
# Llama-3.1-8B: Generate remaining masks + run ALL probe analyses
# ============================================================================
# Submits a chain of SLURM jobs:
#   Job 1: Cold-start masks (DPO + GRPO × snip/cav/fisher) — 1 GPU
#   Job 2: Warm-start GRPO masks (from existing dense GRPO deltas) — 1 GPU
#   Job 3: Probe analysis on ALL mask pairs (after Jobs 1+2) — 1 GPU
#
# Existing resources (already completed):
#   Warm DPO:  /scratch/xie.yiyi/rl_casino_masks/llama8b/warm_{magnitude,momentum}_step50_sp97.5.pt
#   GRPO deltas: /scratch/xie.yiyi/rl_casino_outputs/llama8b_dpo_mask_grpo_dense_baseline_4gpu_20260408_022221/deltas
#
# Usage:
#   bash scripts/run_llama8b_all_masks_and_probes.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# ── Hardcoded paths (from cluster inspection) ───────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"
SPARSITY="${SPARSITY:-97.5}"
N_SAMPLES="${N_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-256}"

SCRATCH_ROOT="/scratch/xie.yiyi"

# Existing warm DPO masks
WARM_DPO_MAGNITUDE="${SCRATCH_ROOT}/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt"
WARM_DPO_MOMENTUM="${SCRATCH_ROOT}/rl_casino_masks/llama8b/warm_momentum_step50_sp97.5.pt"
# Fisher warm DPO — still queued; probes involving it will be skipped until it exists
WARM_DPO_FISHER="${SCRATCH_ROOT}/rl_casino_masks/llama8b/warm_fisher_step50_sp97.5.pt"

# Existing GRPO deltas (1gpu run with deltas up to step 200)
GRPO_DELTA_DIR="${GRPO_DELTA_DIR:-${SCRATCH_ROOT}/rl_casino_outputs/llama8b_dpo_mask_grpo_dense_baseline_1gpu_20260408_044634/deltas/meta_llama_llama_3_1_8b_instruct_math_220k_grpo_dense}"

# New output dirs
COLD_MASK_DIR="${COLD_MASK_DIR:-${SCRATCH_ROOT}/rl_casino_masks/llama8b_cold}"
WARM_GRPO_MASK_DIR="${WARM_GRPO_MASK_DIR:-${SCRATCH_ROOT}/rl_casino_masks/llama8b_warm_grpo}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${SCRATCH_ROOT}/rl_casino_outputs/probe_results_llama8b}"

echo "=============================================="
echo " Llama-8B: All Masks + Probe Analysis"
echo "=============================================="
echo "MODEL:           $MODEL"
echo "SPARSITY:        ${SPARSITY}%"
echo "COLD_MASK_DIR:   $COLD_MASK_DIR"
echo "WARM_GRPO_MASKS: $WARM_GRPO_MASK_DIR"
echo "GRPO_DELTA_DIR:  $GRPO_DELTA_DIR"
echo "PROBE_OUTPUT:    $PROBE_OUTPUT_DIR"
echo "=============================================="

# ════════════════════════════════════════════════════════════════════════════
# Job 1: Cold-start masks (DPO + GRPO × 3 methods = 6 masks)
# ════════════════════════════════════════════════════════════════════════════

JOB1_SCRIPT="${REPO_ROOT}/logs/job1_cold_masks.sh"
cat > "$JOB1_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=llama8b_cold_masks
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/llama8b_cold_masks_%j.out
#SBATCH --error=logs/llama8b_cold_masks_%j.err

set -euo pipefail

source "\$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/\$USER/.conda/envs/rl_casino || conda activate /scratch/\$USER/conda_envs/rl_casino

export HF_HOME="/scratch/\$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/\$USER/hf_cache/datasets"
export PYTHONPATH="\${SLURM_SUBMIT_DIR}:\${PYTHONPATH:-}"

cd "\${SLURM_SUBMIT_DIR}"
mkdir -p "${COLD_MASK_DIR}"

echo "=== Cold-start mask generation (Llama 8B) ==="

for MODE in dpo grpo; do
  for METHOD in snip cav fisher; do
    OUTFILE="${COLD_MASK_DIR}/cold_\${METHOD}_\${MODE}.pt"
    if [ -f "\$OUTFILE" ]; then
      echo "SKIP: \$OUTFILE already exists"
      continue
    fi
    echo ""
    echo "--- Cold \${METHOD} \${MODE} ---"
    python src/cold_start/inference_mask_finder.py \\
      --model_name ${MODEL} \\
      --method "\$METHOD" \\
      --mode "\$MODE" \\
      --n_samples ${N_SAMPLES} \\
      --sparsity ${SPARSITY} \\
      --batch_size ${BATCH_SIZE} \\
      --max_length ${MAX_LENGTH} \\
      --output "\$OUTFILE" || echo "WARNING: cold \${METHOD} \${MODE} failed"
  done
done

echo "=== Cold masks done ==="
ls -la "${COLD_MASK_DIR}/"
EOF

echo "[Job 1] Submitting cold-start masks..."
JOB1_ID=$(sbatch --parsable "$JOB1_SCRIPT")
echo "[Job 1] Cold masks: SLURM job $JOB1_ID"

# ════════════════════════════════════════════════════════════════════════════
# Job 2: Warm-start GRPO masks (from existing GRPO deltas)
# ════════════════════════════════════════════════════════════════════════════

JOB2_SCRIPT="${REPO_ROOT}/logs/job2_warm_grpo_masks.sh"
cat > "$JOB2_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=llama8b_warm_grpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/llama8b_warm_grpo_%j.out
#SBATCH --error=logs/llama8b_warm_grpo_%j.err

set -euo pipefail

source "\$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/\$USER/.conda/envs/rl_casino || conda activate /scratch/\$USER/conda_envs/rl_casino

export PYTHONPATH="\${SLURM_SUBMIT_DIR}:\${PYTHONPATH:-}"
cd "\${SLURM_SUBMIT_DIR}"
mkdir -p "${WARM_GRPO_MASK_DIR}"

GRPO_DELTA_DIR="${GRPO_DELTA_DIR}"
TARGET_STEP=200

echo "=== Warm-start GRPO masks (Llama 8B) ==="
echo "GRPO deltas: \$GRPO_DELTA_DIR"
echo "Available deltas:"
ls "\$GRPO_DELTA_DIR"/*.pt 2>/dev/null || echo "(none found)"

for METHOD in magnitude momentum fisher; do
  OUTFILE="${WARM_GRPO_MASK_DIR}/warm_\${METHOD}_grpo.pt"
  if [ -f "\$OUTFILE" ]; then
    echo "SKIP: \$OUTFILE exists"
    continue
  fi
  echo ""
  echo "--- Warm \${METHOD} GRPO ---"
  python src/warm_start/even_better_mask_finder.py \\
    --delta_log_dir "\$GRPO_DELTA_DIR" \\
    --method "\$METHOD" \\
    --sparsity_percent ${SPARSITY} \\
    --target_step "\$TARGET_STEP" \\
    --min_layer_keep_ratio 0.0025 \\
    --output_file "\$OUTFILE" || echo "WARNING: warm \${METHOD} GRPO failed"
done

echo "=== Warm GRPO masks done ==="
ls -la "${WARM_GRPO_MASK_DIR}/"
EOF

echo ""
echo "[Job 2] Submitting warm-start GRPO masks..."
JOB2_ID=$(sbatch --parsable "$JOB2_SCRIPT")
echo "[Job 2] Warm GRPO masks: SLURM job $JOB2_ID"

# ════════════════════════════════════════════════════════════════════════════
# Job 3: Probe analysis on ALL mask pairs (depends on Jobs 1 + 2)
# ════════════════════════════════════════════════════════════════════════════

JOB3_SCRIPT="${REPO_ROOT}/logs/job3_probe_analysis.sh"
cat > "$JOB3_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=llama8b_probes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/llama8b_probes_%j.out
#SBATCH --error=logs/llama8b_probes_%j.err

set -euo pipefail

source "\$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate /home/\$USER/.conda/envs/rl_casino || conda activate /scratch/\$USER/conda_envs/rl_casino

export HF_HOME="/scratch/\$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/\$USER/hf_cache/datasets"
export PYTHONPATH="\${SLURM_SUBMIT_DIR}:\${PYTHONPATH:-}"
cd "\${SLURM_SUBMIT_DIR}"

MODEL="${MODEL}"
LAYER_STRIDE=4    # Llama 8B has 32 layers
BATCH_SIZE=4
MAX_LENGTH=256    # was 128; bumped to cover AG News / GSM8K tails
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR}"
PROBE_CACHE="\${PROBE_OUTPUT_DIR}/probe_dataset_cache_v2.json"

# Mask directories
COLD="${COLD_MASK_DIR}"
WARM_GRPO="${WARM_GRPO_MASK_DIR}"

# Warm DPO masks (existing)
WARM_DPO_MAG="${WARM_DPO_MAGNITUDE}"
WARM_DPO_MOM="${WARM_DPO_MOMENTUM}"
WARM_DPO_FISH="${WARM_DPO_FISHER}"

mkdir -p "\$PROBE_OUTPUT_DIR"

# Clear stale results from previous runs (different probe samples)
echo "Clearing old probe results for a clean re-run..."
find "\$PROBE_OUTPUT_DIR" -name "probe_results.json" -delete 2>/dev/null || true
rm -f "\$PROBE_CACHE"

echo "=============================================="
echo " Probe Analysis — Llama 8B — ALL mask pairs"
echo "=============================================="

run_probe() {
  local LABEL="\$1" MASK_A="\$2" MASK_B="\$3" LABEL_A="\$4" LABEL_B="\$5"
  local OUT_SUBDIR="\${PROBE_OUTPUT_DIR}/\${LABEL}"

  if [ ! -f "\$MASK_A" ]; then
    echo "SKIP \$LABEL: mask_a missing: \$MASK_A"
    return 0
  fi
  if [ ! -f "\$MASK_B" ]; then
    echo "SKIP \$LABEL: mask_b missing: \$MASK_B"
    return 0
  fi
  if [ -f "\${OUT_SUBDIR}/probe_results.json" ]; then
    echo "SKIP \$LABEL: results already exist"
    return 0
  fi

  echo ""
  echo "=== Probe: \$LABEL ==="
  echo "  A: \$MASK_A (\$LABEL_A)"
  echo "  B: \$MASK_B (\$LABEL_B)"

  mkdir -p "\$OUT_SUBDIR"
  python src/analysis/probe_analysis.py \\
    --model "\$MODEL" \\
    --mask_a "\$MASK_A" \\
    --mask_b "\$MASK_B" \\
    --mask_a_label "\$LABEL_A" \\
    --mask_b_label "\$LABEL_B" \\
    --output_dir "\$OUT_SUBDIR" \\
    --layer_stride "\$LAYER_STRIDE" \\
    --batch_size "\$BATCH_SIZE" \\
    --max_length "\$MAX_LENGTH" \\
    --probe_cache "\$PROBE_CACHE" \\
    --include_baseline || echo "WARNING: probe \$LABEL failed"
}

# ── 1. Cold-start: DPO vs GRPO (per method) ─────────────────────────────────
echo ">>> [1] Cold-start: DPO vs GRPO <<<"
for METHOD in snip cav fisher; do
  run_probe "cold_\${METHOD}_dpo_vs_grpo" \\
    "\${COLD}/cold_\${METHOD}_dpo.pt" \\
    "\${COLD}/cold_\${METHOD}_grpo.pt" \\
    "Cold-\${METHOD^^}-DPO" \\
    "Cold-\${METHOD^^}-GRPO"
done

# ── 2. Warm-start: DPO vs GRPO (per method) ─────────────────────────────────
echo ""
echo ">>> [2] Warm-start: DPO vs GRPO <<<"
run_probe "warm_magnitude_dpo_vs_grpo" \\
  "\$WARM_DPO_MAG" \\
  "\${WARM_GRPO}/warm_magnitude_grpo.pt" \\
  "Warm-Magnitude-DPO" \\
  "Warm-Magnitude-GRPO"

run_probe "warm_momentum_dpo_vs_grpo" \\
  "\$WARM_DPO_MOM" \\
  "\${WARM_GRPO}/warm_momentum_grpo.pt" \\
  "Warm-Momentum-DPO" \\
  "Warm-Momentum-GRPO"

run_probe "warm_fisher_dpo_vs_grpo" \\
  "\$WARM_DPO_FISH" \\
  "\${WARM_GRPO}/warm_fisher_grpo.pt" \\
  "Warm-Fisher-DPO" \\
  "Warm-Fisher-GRPO"

# ── 3. Cold vs Warm within DPO ──────────────────────────────────────────────
echo ""
echo ">>> [3] Cold vs Warm (DPO) <<<"
run_probe "cold_vs_warm_fisher_dpo" \\
  "\${COLD}/cold_fisher_dpo.pt" \\
  "\$WARM_DPO_FISH" \\
  "Cold-Fisher-DPO" \\
  "Warm-Fisher-DPO"

# ── 4. Cold vs Warm within GRPO ─────────────────────────────────────────────
echo ""
echo ">>> [4] Cold vs Warm (GRPO) <<<"
run_probe "cold_vs_warm_fisher_grpo" \\
  "\${COLD}/cold_fisher_grpo.pt" \\
  "\${WARM_GRPO}/warm_fisher_grpo.pt" \\
  "Cold-Fisher-GRPO" \\
  "Warm-Fisher-GRPO"

# ── 5. Cross-method within cold DPO ─────────────────────────────────────────
echo ""
echo ">>> [5] Cross-method (cold DPO) <<<"
run_probe "cold_dpo_snip_vs_fisher" \\
  "\${COLD}/cold_snip_dpo.pt" \\
  "\${COLD}/cold_fisher_dpo.pt" \\
  "Cold-SNIP-DPO" \\
  "Cold-Fisher-DPO"

run_probe "cold_dpo_snip_vs_cav" \\
  "\${COLD}/cold_snip_dpo.pt" \\
  "\${COLD}/cold_cav_dpo.pt" \\
  "Cold-SNIP-DPO" \\
  "Cold-CAV-DPO"

# ── 6. Cross-method within cold GRPO ────────────────────────────────────────
echo ""
echo ">>> [6] Cross-method (cold GRPO) <<<"
run_probe "cold_grpo_snip_vs_fisher" \\
  "\${COLD}/cold_snip_grpo.pt" \\
  "\${COLD}/cold_fisher_grpo.pt" \\
  "Cold-SNIP-GRPO" \\
  "Cold-Fisher-GRPO"

run_probe "cold_grpo_snip_vs_cav" \\
  "\${COLD}/cold_snip_grpo.pt" \\
  "\${COLD}/cold_cav_grpo.pt" \\
  "Cold-SNIP-GRPO" \\
  "Cold-CAV-GRPO"

echo ""
echo "=============================================="
echo " ALL probe analyses complete!"
echo " Results in: \$PROBE_OUTPUT_DIR/"
echo "=============================================="
ls -d "\$PROBE_OUTPUT_DIR"/*/ 2>/dev/null || echo "(no subdirectories)"
EOF

echo ""
echo "[Job 3] Submitting probe analysis (depends on Jobs 1 + 2)..."
JOB3_ID=$(sbatch --parsable --dependency=afterok:${JOB1_ID}:${JOB2_ID} "$JOB3_SCRIPT")
echo "[Job 3] Probe analysis: SLURM job $JOB3_ID"

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo " ALL JOBS SUBMITTED"
echo "=============================================="
echo ""
echo "  Job 1 (cold masks):       $JOB1_ID  [independent]"
echo "  Job 2 (warm GRPO masks):  $JOB2_ID  [independent]"
echo "  Job 3 (probe analysis):   $JOB3_ID  [after Jobs 1+2]"
echo ""
echo "Probe comparisons that will run:"
echo "  - Cold DPO vs GRPO       (snip, cav, fisher)"
echo "  - Warm DPO vs GRPO       (magnitude, momentum, fisher)"
echo "  - Cold vs Warm DPO       (fisher)"
echo "  - Cold vs Warm GRPO      (fisher)"
echo "  - Cross-method cold DPO  (snip vs fisher, snip vs cav)"
echo "  - Cross-method cold GRPO (snip vs fisher, snip vs cav)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f logs/llama8b_cold_masks_${JOB1_ID}.out"
echo "=============================================="

#!/bin/bash
# Random-mask control for the Qwen3-32B chain: density-matched to the p2 oracle
# mask @97.5% (same selector path via random_mask_baseline.py, seed 42 — the
# paper's standard random seed, matching the Llama random baseline).
#
# Memory: reference masks ~33 GB + float64 uniform scores ~262 GB + selector
# overhead -> 800G (matches p2 oracle). float64 scores are REQUIRED at this
# scale: job 248909 failed with float32 (boundary tie collisions in the
# chunked selector, ~1950 duplicates per representable value at the threshold).
#
# Usage: sbatch --chdir=$W/rc-sparse-speed scripts/aicr_qwen32b_mask_random.sbatch
#SBATCH --job-name=q32b_mask_rand
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=08:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/q32b_mask_rand_%j.out

set -euo pipefail
cd /work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

SP=97.5
FLOOR=0.0025
SEED=42
REF=$S/transfer_v1/oracle_masks_qwen3_32b/oracle_dpo_light_r1_step500_sp${SP}.pt
OUT=$S/transfer_v1/oracle_masks_qwen3_32b/random_baseline_light_r1_sp${SP}_seed${SEED}.pt

if [ ! -f "$REF" ]; then echo "FATAL: reference oracle mask missing: $REF"; exit 1; fi
if [ -s "$OUT" ]; then echo "mask already exists: $OUT"; exit 0; fi

echo "=== Qwen3-32B random baseline mask | sp=${SP} seed=${SEED} | job ${SLURM_JOB_ID:-local} | $(date) ==="
python src/warm_start/random_mask_baseline.py \
    --reference_mask "$REF" \
    --sparsity_percent $SP \
    --min_layer_keep_ratio $FLOOR \
    --seed $SEED \
    --float64_scores \
    --compare_to_reference \
    --output_file "$OUT"

echo "=== density check ==="
python - "$OUT" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = d["masks"] if isinstance(d, dict) and "masks" in d else d
kept = sum(int(t.sum()) for t in m.values()); tot = sum(t.numel() for t in m.values())
print(f"random mask: tensors={len(m)} kept={kept:,}/{tot:,} ({100*kept/tot:.4f}% density)")
PY
ls -lh "$OUT"
echo "Finished: $(date)"

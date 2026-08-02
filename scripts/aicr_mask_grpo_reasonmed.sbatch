#!/bin/bash
# Oracle mask for the GRPO ReasonMed dense run — checkpoint-diff route, identical
# construction to the paper's oracle_grpo_math220k_step500_sp97.5.pt (2026-04-29),
# so the Figure-2 GRPO⇄GRPO pair compares masks built the same way. Same defaults
# as every other oracle in this sweep: global pooling, min_layer_keep_ratio 0.0025
# (the script default), --force_cpu, rho=97.5.
#
# Chained: submit with --dependency=afterok:<reasonmed dense job>.
# CPU/memory-latency bound; measured checkpoint-diff peak at rho=97.5 was 105 GB,
# and the selector is markedly slower on first-gen EPYC (oracle_sparsity_sweep_
# lightr1_2026-07-28.md §9), hence the generous cpu/mem/time ask.
#
# Usage: sbatch --dependency=afterok:243101 aicr_mask_grpo_reasonmed.sbatch
#SBATCH --job-name=rmed_mask
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/rmed_mask_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

CKPT="$S/rebuttal_analysis/grpo_matched_aicr/dense_reasonmed/grpo_reasonmed_500steps_cap2048/checkpoints/checkpoint-500"
OUT="$S/transfer_v1/oracle_masks_llama8b/oracle_grpo_reasonmed_step500_sp97.5.pt"

[ -d "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
[ -f "$OUT" ] && { echo "Mask already exists: $OUT"; exit 0; }

echo "=========================================="
echo "GRPO ReasonMed oracle mask | checkpoint-diff | rho=97.5"
echo "  ckpt=$CKPT"
echo "  out=$OUT"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

cd /work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed
python src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model meta-llama/Llama-3.1-8B-Instruct \
    --final_model "$CKPT" \
    --sparsity_percent 97.5 \
    --force_cpu \
    --output_file "$OUT"

echo "Finished: $(date)"

# sacct COMPLETED is not proof the work happened (DO_NOT_REPEAT) — verify density,
# and verify the tensor set matches the math mask or the Jaccard is meaningless.
python - "$OUT" "$S/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt" <<'PY'
import sys, torch
def load(p):
    m = torch.load(p, map_location="cpu", weights_only=False)
    return m.get("masks", m)
new = load(sys.argv[1])
kept = sum(v.sum().item() for v in new.values())
tot = sum(v.numel() for v in new.values())
print(f"kept {kept:,} / {tot:,} = {kept/tot:.4%} (target 2.5000%)")
try:
    ref = load(sys.argv[2])
except Exception as e:
    print(f"(math mask not loadable here: {e})"); raise SystemExit
a, b = set(new), set(ref)
print(f"tensors: new {len(a)}  math {len(b)}  shared {len(a & b)}")
if a != b:
    print(f"WARNING: tensor sets differ — new-only {sorted(a-b)[:5]}, math-only {sorted(b-a)[:5]}")
inter = sum((new[k] & ref[k]).sum().item() for k in a & b)
union = sum((new[k] | ref[k]).sum().item() for k in a & b)
print(f"GLOBAL Jaccard(ReasonMed, OpenR1-Math) = {inter/union:.4f}   (chance at 2.5% ~= 1.27e-2)")
PY

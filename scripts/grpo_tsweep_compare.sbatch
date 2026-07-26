#!/bin/bash
# Compare the GRPO warm-start magnitude masks at T = 50 / 100 / 250 against the
# oracle (T = 500), on the trajectory built by scripts/grpo_tsweep_dense.sbatch.
#
# THE QUESTION: how early in a GRPO run is the final subnetwork already decided?
# The paper answers this for DPO (warm-magnitude k = 50/100/150/200 vs oracle) and
# never answers it for GRPO. Reviewer bf6h's W1 and AC item 3 both live here.
#
# WHAT IS MEASURED
#   Jaccard, global and per-layer  — overlap of the retained coordinate sets.
#   Linear CKA                     — whether two masks select functionally similar
#                                    subspaces even where they disagree coordinate-wise.
#   Tensor-class composition       — which KINDS of tensor each T selects, the axis
#                                    from docs/run_logs/mask_composition_2026-07-25.md.
#
# TWO CONTROLS ARE IN THE MASK SET ON PURPOSE
#   Random-sp97.5   — the chance floor. App. A.6.1 gives E[J] = A/(2-A) ~= 1.27e-2 at
#                     rho = 97.5%, so any T-vs-oracle Jaccard must be read against it.
#   Oracle-paper-s500 — the GRPO oracle already in the paper, built from the OLD dense
#                     run (cosine over max_steps=5000, so a 500-step warmup) with the
#                     same mask-finder code. New-oracle vs paper-oracle is therefore a
#                     clean read on how much of the "oracle subnetwork" is an artifact
#                     of the LR schedule rather than of the objective.
#
# GPU is genuinely used here: CKA runs forward passes over a calibration set.
#SBATCH --job-name=grpo_tsweep_cmp
#SBATCH --output=logs/grpo_tsweep_cmp_%j.out
#SBATCH --error=logs/grpo_tsweep_cmp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=05:00:00

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO" || exit 1
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_USER_ROOT:-/scratch/$USER}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"

SPARSITY="${SPARSITY:-97.5}"
M="$SCRATCH_ROOT/rebuttal_analysis/grpo_tsweep/masks"
OLD="$SCRATCH_ROOT/transfer_v1/oracle_masks_llama8b"
OUT="$SCRATCH_ROOT/rebuttal_analysis/grpo_tsweep/compare"
mkdir -p "$OUT"

MASKS_JSON="$OUT/masks.json"
cat > "$MASKS_JSON" << JSONEOF
[
  {"label": "GRPO-warmmag-T50",     "family": "GRPO-warmmag", "dataset": "Math220k", "step": 50,  "sparsity": $SPARSITY,
   "path": "$M/grpo_math220k_warmmag_T50_sp${SPARSITY}.pt"},
  {"label": "GRPO-warmmag-T100",    "family": "GRPO-warmmag", "dataset": "Math220k", "step": 100, "sparsity": $SPARSITY,
   "path": "$M/grpo_math220k_warmmag_T100_sp${SPARSITY}.pt"},
  {"label": "GRPO-warmmag-T150",    "family": "GRPO-warmmag", "dataset": "Math220k", "step": 150, "sparsity": $SPARSITY,
   "path": "$M/grpo_math220k_warmmag_T150_sp${SPARSITY}.pt"},
  {"label": "GRPO-warmmag-T200",    "family": "GRPO-warmmag", "dataset": "Math220k", "step": 200, "sparsity": $SPARSITY,
   "path": "$M/grpo_math220k_warmmag_T200_sp${SPARSITY}.pt"},
  {"label": "GRPO-warmmag-T250",    "family": "GRPO-warmmag", "dataset": "Math220k", "step": 250, "sparsity": $SPARSITY,
   "path": "$M/grpo_math220k_warmmag_T250_sp${SPARSITY}.pt"},
  {"label": "GRPO-oracle-T500",     "family": "GRPO-oracle",  "dataset": "Math220k", "step": 500, "sparsity": $SPARSITY,
   "path": "$M/grpo_math220k_oracle_T500_sp${SPARSITY}.pt"},
  {"label": "GRPO-oracle-paper500", "family": "GRPO-oracle",  "dataset": "Math220k", "step": 500, "sparsity": $SPARSITY,
   "path": "$OLD/oracle_grpo_math220k_step500_sp${SPARSITY}.pt"},
  {"label": "Random-seed42",        "family": "random",       "dataset": "n/a",      "step": null, "sparsity": $SPARSITY,
   "path": "$OLD/random_baseline_lightr1_sp${SPARSITY}_seed42.pt"}
]
JSONEOF

echo "=========================================="
echo "GRPO T-sweep mask comparison"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
cat "$MASKS_JSON"
echo "=========================================="

MISSING=0
for p in $(python -c "import json;[print(e['path']) for e in json.load(open('$MASKS_JSON'))]"); do
    [ -f "$p" ] || { echo "FATAL: missing mask $p"; MISSING=1; }
done
[ "$MISSING" -ne 0 ] && exit 1

echo ""
echo "[1/2] Jaccard (global + per-layer) and linear CKA over all 28 pairs"
python scripts/multi_mask_jaccard_cka.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUT" \
    --layer_substr down_proj \
    --n_samples 64 \
    --max_length 512 \
    --batch_size 4 \
    --seed 42 \
    --calibration_dataset tulu3
RC1=$?

echo ""
echo "[2/2] Tensor-class composition, same convention as mask_composition_2026-07-25"
python src/analysis/mask_composition.py \
    "$M/grpo_math220k_warmmag_T50_sp${SPARSITY}.pt" \
    "$M/grpo_math220k_warmmag_T100_sp${SPARSITY}.pt" \
    "$M/grpo_math220k_warmmag_T150_sp${SPARSITY}.pt" \
    "$M/grpo_math220k_warmmag_T200_sp${SPARSITY}.pt" \
    "$M/grpo_math220k_warmmag_T250_sp${SPARSITY}.pt" \
    "$M/grpo_math220k_oracle_T500_sp${SPARSITY}.pt" \
    "$OLD/oracle_grpo_math220k_step500_sp${SPARSITY}.pt" \
    "$OLD/random_baseline_lightr1_sp${SPARSITY}_seed42.pt" \
    --out "$OUT/mask_composition_grpo_tsweep.json"
RC2=$?

echo ""
echo "=========================================="
ls -lh "$OUT" 2>&1
echo "jaccard_cka exit=$RC1  composition exit=$RC2  finished $(date)"
echo "=========================================="
[ $RC1 -ne 0 ] && exit $RC1
exit $RC2

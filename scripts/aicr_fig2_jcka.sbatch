#!/bin/bash
# Figure-2 revision data: pairwise per-layer Jaccard + CKA over five masks,
# adding the GRPO⇄GRPO cell (paper GRPO-OpenR1 mask vs the new GRPO-Tülu3RLVR mask).
# Paper-era masks are used for every pre-existing pair so the DPO⇄DPO and DPO⇄GRPO
# lines share provenance with the submitted figure; only the Tülu3-RLVR mask is new.
#
# Chained: submit with --dependency=afterok:<t3rlvr_mask job>.
# MD5 gates on the two Explorer-pushed masks fail the job fast if a transfer was
# incomplete — sacct COMPLETED on the push is not proof (DO_NOT_REPEAT).
#
# Usage: sbatch --dependency=afterok:<maskjob> aicr_fig2_jcka.sbatch
#SBATCH --job-name=fig2_jcka
# b200-batch, not b200-devel: the devel QOS caps a user at gres/gpu=2 total, and the
# Qwen3-32B p1 chain lives there — one 2-GPU q32b job saturates it and this lands in
# QOSMaxGRESPerUser behind a multi-day chain. batch allows 32.
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/fig2_jcka_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

# The CKA calibration set (`tulu3` = allenai/llama-3.1-tulu-3-8b-preference-mixture,
# 272,898 rows) is fetched at runtime by mask_to_cka.load_calibration_samples and had
# never been cached on AICR — 239138 died on OfflineModeIsEnabled. It was pre-fetched
# on the login node 2026-07-30, so aicr_env.sh's HF_*_OFFLINE=1 can stay as-is; this
# job now resolves it from $HF_DATASETS_CACHE with no network.

MASK_DIR="$S/transfer_v1/oracle_masks_llama8b"
OUT_DIR="$S/rebuttal_analysis/fig2_jcka_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

declare -A WANT_MD5=(
  ["$MASK_DIR/oracle_dpo_lightr1_step500_sp97.5_paper.pt"]="4d8b5cc255d8c31e0815766fa4121a3a"
  ["$MASK_DIR/oracle_dpo_tulu3_step500_sp97.5.pt"]="9d0d8af824327778d1b8c88f0d6bebc2"
)
for f in "${!WANT_MD5[@]}"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
  got=$(md5sum "$f" | cut -d' ' -f1)
  [ "$got" = "${WANT_MD5[$f]}" ] || { echo "FATAL: md5 mismatch $f ($got)"; exit 1; }
  echo "md5 OK  $f"
done
for f in "$MASK_DIR/oracle_grpo_math220k_step500_sp97.5.pt" \
         "$MASK_DIR/oracle_grpo_tulu3rlvr_step500_sp97.5.pt" \
         "$MASK_DIR/random_baseline_lightr1_sp97.5_seed42.pt"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

cat > "$OUT_DIR/masks.json" <<EOF
[
  {"label": "DPO-LightR1",    "path": "$MASK_DIR/oracle_dpo_lightr1_step500_sp97.5_paper.pt", "family": "DPO-oracle",  "dataset": "LightR1",   "step": 500, "sparsity": 97.5},
  {"label": "DPO-Tulu3",      "path": "$MASK_DIR/oracle_dpo_tulu3_step500_sp97.5.pt",         "family": "DPO-oracle",  "dataset": "Tulu3",     "step": 500, "sparsity": 97.5},
  {"label": "GRPO-OpenR1",    "path": "$MASK_DIR/oracle_grpo_math220k_step500_sp97.5.pt",     "family": "GRPO-oracle", "dataset": "OpenR1",    "step": 500, "sparsity": 97.5},
  {"label": "GRPO-Tulu3RLVR", "path": "$MASK_DIR/oracle_grpo_tulu3rlvr_step500_sp97.5.pt",    "family": "GRPO-oracle", "dataset": "Tulu3RLVR", "step": 500, "sparsity": 97.5},
  {"label": "Random-s42",     "path": "$MASK_DIR/random_baseline_lightr1_sp97.5_seed42.pt",   "family": "random",      "dataset": "LightR1",   "step": 0,   "sparsity": 97.5}
]
EOF

echo "=========================================="
echo "Figure 2 revision — 5-mask pairwise Jaccard + CKA"
echo "  out=$OUT_DIR"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

cd /work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed
python scripts/multi_mask_jaccard_cka.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --masks_json "$OUT_DIR/masks.json" \
    --output_dir "$OUT_DIR" \
    --n_samples 64 \
    --max_length 512 \
    --batch_size 4 \
    --seed 42

echo "Finished: $(date)"
ls -la "$OUT_DIR"

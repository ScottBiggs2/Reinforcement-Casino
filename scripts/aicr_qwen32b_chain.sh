#!/bin/bash
# Submit the whole Qwen3-32B chain on AICR as one afterok dependency chain:
#   p1 x3 (dense resume, 24h slots) -> p2 (oracle mask) -> p3 x3 (sparse, 24h slots)
# Extra training slots are free: once a phase's checkpoint-500 exists the next
# slot exits 0 in seconds. A hard crash (rc not in {0,124}) breaks the chain so
# nothing trains on top of a corrupt state.
#
# Run on the AICR login node AFTER the checkpoint-250 transfer is md5-verified
# and CKPT250_VERIFIED is in place:
#   bash /work/neu/p2026_0038_neu/xie_yiyi/aicr_qwen32b_chain.sh
set -euo pipefail
W=/work/neu/p2026_0038_neu/xie_yiyi
cd "$W"

GATE=/scratch/xie_yiyi_neu/transfer_v1/dense_dpo_light_r1_qwen3_32b/CKPT250_VERIFIED
if [ ! -f "$GATE" ]; then
  echo "FATAL: $GATE missing — verify the checkpoint-250 transfer first."; exit 1
fi

# Keep the GPU slots off nodes already running our own jobs (memory-bandwidth
# contention lesson: 235302 on b0003 ran at 156 s/it vs 36.5 beside n5_masks).
# EXCLUDE_NODES may be overridden at invocation; default = nodes busy at submit.
EXCLUDE_NODES="${EXCLUDE_NODES:-$(squeue -u "$USER" -h -t RUNNING -o %N | sort -u | paste -sd, -)}"
EXCL=""
[ -n "$EXCLUDE_NODES" ] && EXCL="--exclude=$EXCLUDE_NODES"
echo "GPU slots exclude: ${EXCLUDE_NODES:-none}"

P1A=$(sbatch --parsable $EXCL "$W/aicr_qwen32b_p1_dense.sbatch")
P1B=$(sbatch --parsable $EXCL --dependency=afterok:$P1A "$W/aicr_qwen32b_p1_dense.sbatch")
P1C=$(sbatch --parsable $EXCL --dependency=afterok:$P1B "$W/aicr_qwen32b_p1_dense.sbatch")
P2=$(sbatch  --parsable --dependency=afterok:$P1C "$W/aicr_qwen32b_p2_oracle.sbatch")
P3A=$(sbatch --parsable $EXCL --dependency=afterok:$P2  "$W/aicr_qwen32b_p3_sparse.sbatch")
P3B=$(sbatch --parsable $EXCL --dependency=afterok:$P3A "$W/aicr_qwen32b_p3_sparse.sbatch")
P3C=$(sbatch --parsable $EXCL --dependency=afterok:$P3B "$W/aicr_qwen32b_p3_sparse.sbatch")

echo "chain: p1 $P1A -> $P1B -> $P1C -> p2 $P2 -> p3 $P3A -> $P3B -> $P3C"

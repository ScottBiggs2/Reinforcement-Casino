#!/bin/bash
# Launcher: full faithful high-sparsity DPO chain on Qwen/Qwen3-8B (instruct),
# for both light-r1 and tulu3, as two parallel dependency chains:
#   p1 dense DPO  --afterok-->  p2 oracle mask @97.5%  --afterok-->  p3 sparse DPO
# The step-500 oracle mask path is deterministic, so p3 can be queued up front.
# Run from repo root on the cluster:  bash scripts/qwen3_transfer_launch.sh
set -euo pipefail
cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs
S=scripts
ORACLE_DIR=/scratch/xie.yiyi/transfer_v1/oracle_masks_qwen3_8b

jid_of() { grep -oE '[0-9]+$' <<<"$1"; }

for DS in light-r1 tulu3; do
  DS_TAG=$(echo "$DS" | tr '-' '_')
  MASK="$ORACLE_DIR/oracle_dpo_${DS_TAG}_step500_sp97.5.pt"

  P1=$(jid_of "$(sbatch --export=ALL,DATASET=$DS "$S/qwen3_p1_dense_dpo.sbatch")")
  echo "[$DS] p1 dense DPO        -> $P1"

  P2=$(jid_of "$(sbatch --dependency=afterok:$P1 --export=ALL,DATASET=$DS "$S/qwen3_p2_oracle_dpo.sbatch")")
  echo "[$DS] p2 oracle mask      -> $P2  (afterok $P1)"

  P3=$(jid_of "$(sbatch --dependency=afterok:$P2 \
        --export=ALL,DATASET=$DS,MASK_PATH=$MASK,RUN_TAG=oracle_step500 \
        "$S/qwen3_p3_sparse_dpo.sbatch")")
  echo "[$DS] p3 sparse DPO       -> $P3  (afterok $P2)  mask=$MASK"
  echo ""
done

echo "All chains queued. Watch with: squeue -u \$USER"

#!/bin/bash
# Diagnose the exactly-0.0 CKA between the two GRPO oracle masks (job 240279).
# b200-batch, not b200-devel: the devel QOS caps a user at gres/gpu=2 and the
# Qwen3-32B chain occupies it.
#
# Usage: sbatch aicr_diag_cka_zero.sbatch
#SBATCH --job-name=diag_cka0
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/diag_cka0_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

MASK_DIR="$S/transfer_v1/oracle_masks_llama8b"
cd /work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed

python scripts/diag_cka_zero_pair.py \
    "$MASK_DIR/oracle_grpo_math220k_step500_sp97.5.pt" \
    "$MASK_DIR/oracle_grpo_tulu3rlvr_step500_sp97.5.pt"

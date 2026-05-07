#!/bin/bash
# Reconstruct deltas_step_{100,150,200}.pt for Light-R1 dense DPO from
# already-saved HF checkpoints + base_state.pt. Cheap — no training needed.
#SBATCH --job-name=synth_lightr1_deltas
#SBATCH --output=logs/synth_lightr1_deltas_%j.out
#SBATCH --error=logs/synth_lightr1_deltas_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"

DELTA_DIR="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/deltas/meta_llama_llama_3_1_8b_instruct_light_r1"
CKPT_ROOT="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1"

echo "=========================================="
echo "Synthesize Light-R1 deltas at steps 100/150/200"
echo "BASE_STATE = $DELTA_DIR/base_state.pt"
echo "CKPT_ROOT  = $CKPT_ROOT"
echo "OUT_DIR    = $DELTA_DIR"
echo "Started: $(date)"
echo "=========================================="

python scripts/synth_deltas_from_ckpts.py \
    --base_state "$DELTA_DIR/base_state.pt" \
    --ckpt_root "$CKPT_ROOT" \
    --steps 100 150 200 \
    --out_dir "$DELTA_DIR"

echo ""
echo "=== Final delta listing ==="
ls -lh "$DELTA_DIR/"deltas_step_*.pt
echo ""
echo "Finished: $(date)"

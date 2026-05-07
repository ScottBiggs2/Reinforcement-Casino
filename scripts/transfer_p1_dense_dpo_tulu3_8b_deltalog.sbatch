#!/bin/bash
# Re-run dense DPO on Tulu3 to capture delta logs at steps {50,100,150,200}
# (the original run only logged step 50 because DPO_train default capped
# delta_log_end_step at 50 for num_steps=500). All hyperparameters identical
# to scripts/transfer_p1_dense_dpo_tulu3_8b.sbatch — only delta logging knobs
# and OUT_BASE differ.
#SBATCH --job-name=transfer_p1_dpo_tulu3_dlog
#SBATCH --output=logs/transfer_p1_dpo_tulu3_dlog_%j.out
#SBATCH --error=logs/transfer_p1_dpo_tulu3_dlog_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
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
export HF_DATASETS_CACHE="/scratch/xie.yiyi/hf_cache/datasets"
export WANDB_PROJECT="rl_casino_transfer_v1"

# NEW output dir to avoid clobbering existing dense_dpo_tulu3_llama8b artifacts.
OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b_deltalog"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)
echo "Visible GPUs: $NGPUS"

echo "=========================================="
echo "Phase 1B-redo: Dense DPO with delta logging  |  Llama-3.1-8B-Instruct  |  Tulu3"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "Same hparams as transfer_p1_dense_dpo_tulu3_8b.sbatch."
echo "Adds: --delta_log_interval 50 --delta_log_end_step 200 (Scott pipeline_common.sh defaults)."
echo "Output dir is _deltalog suffix to keep original ckpts intact."
echo ""

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/DPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset tulu3 \
    --num_steps 500 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --save_steps 50 \
    --save_total_limit 8 \
    --delta_log_interval 50 \
    --delta_log_end_step 200 \
    --gradient_checkpointing \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_name dense_dpo_llama8b_tulu3_scott_deltalog

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -5
DELTA_DIR="$OUT_BASE/deltas/meta_llama_llama_3_1_8b_instruct_tulu3"
echo "Delta files at: $DELTA_DIR"
ls -lh "$DELTA_DIR/"deltas_step_*.pt 2>/dev/null
echo "=========================================="

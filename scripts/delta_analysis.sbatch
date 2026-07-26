#!/bin/bash
# Update-vector analysis on checkpoints we already hold. CPU-only, `short` partition.
#
# Runs here rather than on a GPU deliberately: multigpu has exactly ONE h200 node and
# everything requesting it is 8+ hours out tonight, while `short` has ~18 idle nodes and
# starts in minutes. Neither analysis needs a GPU.
#
# ARM 0/1 — DPO vs GRPO, same Llama-3.1-8B base, different objective.
#   Produces the fraction of ||dtheta||^2 held by the top-k% of coordinates for each
#   objective. This is what turns "GRPO tolerates less sparsity than DPO" from a
#   post-hoc relabelling of Appendix D.4's failure into a prediction read off a measured
#   curve, with both objectives on one axis. Mukherjee et al. estimate GRPO's natural
#   update sparsity at ~70% versus DPO's much higher; this measures it on our own runs.
#   Caveat to carry into any write-up: the DPO arm here is the Tulu3 "long" run, because
#   the Llama Light-R1 dense checkpoint is no longer on /scratch.
#
# ARM 2 — Qwen3-8B Light-R1 dense, the strong-signal DPO arm (20.9 epochs, margin 3.24),
#   as a second point on the same axis and a check that the curve is not Tulu3-specific.
#
# ALL ARMS also emit per-matrix stable rank of dW = ||dW||_F^2 / ||dW||_2^2, which LoRA
# caps at r by construction. If the sparse update has stable rank far above 64 at a
# matched trainable-parameter budget, that is the mechanistic answer to "why not just
# use LoRA" — and unlike a downstream accuracy row it survives the two methods tying.
#SBATCH --job-name=delta_analysis
#SBATCH --output=logs/delta_analysis_%A_%a.out
#SBATCH --error=logs/delta_analysis_%A_%a.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
# ~40-50G actual peak (one fp32 8B dict + streamed deltas); 96G matches the profile
# that scheduled instantly for the mask audit. A 256G ask sat in the queue instead.
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO"; mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_USER_ROOT:-/scratch/$USER}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"
export OMP_NUM_THREADS=8

T="$SCRATCH_ROOT/transfer_v1"
OUT="$SCRATCH_ROOT/rebuttal_analysis"
mkdir -p "$OUT"

LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen3-8B"

# tag | initial (theta^0) | final (theta_T)
CFG=(
  "llama_dpo_tulu3|$LLAMA|$T/dense_dpo_tulu3_long_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
  "llama_grpo_math220k|$LLAMA|$T/dense_grpo_math220k_llama8b/llama8b_math220k_dense_scott_full/checkpoints/checkpoint-500"
  "qwen3_dpo_lightr1|$QWEN|$T/dense_dpo_light_r1_qwen3_8b/checkpoints/qwen_qwen3_8b_light_r1/checkpoint-500"
)

IFS='|' read -r TAG INIT FINAL <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "delta analysis: $TAG"
echo "  theta^0 : $INIT"
echo "  theta_T : $FINAL"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

if [ ! -d "$FINAL" ]; then echo "SKIP: missing $FINAL"; exit 0; fi

python src/analysis/delta_analysis.py \
    --initial "$INIT" \
    --final "$FINAL" \
    --fractions 0.0025,0.01,0.025,0.05,0.10,0.25 \
    --stable_rank \
    --rank_max_tensors 48 \
    --tag "$TAG" \
    --out "$OUT/delta_${TAG}.json"
RC=$?

# Propagate the python exit code. Without this the trailing echo made the job report
# COMPLETED 0:0 even when the analysis had thrown — three arms "succeeded" having
# produced nothing.
if [ "$RC" -ne 0 ]; then
    echo "FAILED $TAG rc=$RC: $(date)"
    exit "$RC"
fi
echo "Finished $TAG: $(date)"

#!/usr/bin/env bash
# Full mask interpretation for GRPO Llama 3.1 8B masks: pairwise Jaccard (component + total),
# CKA, effective rank, sparsity CSVs, heatmap, plot_layer_metrics (MC Jaccard), and per-mask linear probes.
#
# Default paths match Explorer scratch layout from your 5way + oracle + random masks (97.5% sparsity).
# Override any path with env vars before sbatch.
#
# Drop-in (repo root on cluster):
#   export HF_TOKEN="${HF_TOKEN:?need Llama HF access}"
#   sbatch scripts/sbatch_grpo_llama31_mask_interpretation_suite.sh
#
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_mask_suite
#SBATCH --output=logs/grpo_mask_interpretation_suite_%j.out
#SBATCH --error=logs/grpo_mask_interpretation_suite_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

mkdir -p logs

SCR="${RL_CASINO_SCRATCH_ROOT:-/scratch/${USER}}"
GRPO_MASK_ROOT="${GRPO_MASK_ROOT:-${SCR}/rl_casino_grpo/masks}"

# Magnitude step 200 (5way run), oracle checkpoint-500 diff, random block 97.5% — from your Explorer session.
MAG="${GRPO_MASK_MAG:-${GRPO_MASK_ROOT}/5way_6512509/grpo_magnitude_meta_llama_llama_3_1_8b_instruct_OpenR1-Math-220k_sp97_5_step200.pt}"
ORACLE="${GRPO_MASK_ORACLE:-${GRPO_MASK_ROOT}/oracle_6518569/checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct.pt}"
RAND="${GRPO_MASK_RANDOM:-${GRPO_MASK_ROOT}/grpo_random_block_sp975_meta_llama_llama_3_1_8b_instruct_OpenR1-Math-220k_block256.pt}"

for f in "$MAG" "$ORACLE" "$RAND"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing mask file: $f" >&2
    echo "Set GRPO_MASK_MAG, GRPO_MASK_ORACLE, GRPO_MASK_RANDOM or GRPO_MASK_ROOT to fix paths." >&2
    exit 2
  fi
done

LIST_FILE="${GRPO_MASK_SUITE_LIST_FILE:-${SCR}/rl_casino_grpo/mask_suite_lists/grpo_mag_oracle_random_${SLURM_JOB_ID:-local}.txt}"
mkdir -p "$(dirname "$LIST_FILE")"
{
  echo "# auto-generated for mask interpretation suite"
  echo "$MAG"
  echo "$ORACLE"
  echo "$RAND"
} >"$LIST_FILE"

export MASK_SUITE_LIST_FILE="$LIST_FILE"
export MASK_SUITE_OUT_DIR="${MASK_SUITE_OUT_DIR:-${SCR}/rl_casino_grpo/mask_interpretation_suite/grpo_llama31_${SLURM_JOB_ID:-local}}"
export MASK_SUITE_LABELS="${MASK_SUITE_LABELS:-mag_step200 oracle_gt random_block97_5}"
export MASK_SUITE_DEVICE="${MASK_SUITE_DEVICE:-cpu}"
export MASK_SUITE_EXTENDED="${MASK_SUITE_EXTENDED:-both}"
export MASK_SUITE_SKIP_EFFECTIVE_RANK="${MASK_SUITE_SKIP_EFFECTIVE_RANK:-0}"
export MASK_SUITE_HEATMAP="${MASK_SUITE_HEATMAP:-1}"
export MASK_SUITE_RUN_CKA="${MASK_SUITE_RUN_CKA:-1}"
export MASK_SUITE_CKA_MODEL="${MASK_SUITE_CKA_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export MASK_SUITE_CKA_DATASET="${MASK_SUITE_CKA_DATASET:-tulu3}"
export MASK_SUITE_CKA_DEVICE="${MASK_SUITE_CKA_DEVICE:-cuda}"
export MASK_SUITE_CKA_N_SAMPLES="${MASK_SUITE_CKA_N_SAMPLES:-64}"
export MASK_SUITE_CKA_BATCH_SIZE="${MASK_SUITE_CKA_BATCH_SIZE:-2}"
export MASK_SUITE_PROBE_REPORTS="${MASK_SUITE_PROBE_REPORTS:-1}"
export MASK_SUITE_PROBE_MODE="${MASK_SUITE_PROBE_MODE:-grpo}"
export MASK_SUITE_RUN_PLOTS="${MASK_SUITE_RUN_PLOTS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "LIST_FILE=$LIST_FILE"
echo "OUT_DIR=$MASK_SUITE_OUT_DIR"
echo "LABELS=$MASK_SUITE_LABELS"

exec bash "${REPO_ROOT}/scripts/sbatch_mask_interpretation_suite.sh"

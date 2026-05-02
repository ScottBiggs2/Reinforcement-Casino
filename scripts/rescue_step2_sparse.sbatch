#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --job-name=sp_launch
#SBATCH --output=logs/rescue_launch_%j.out

set -e
export PIPELINE_RUN_ID="tulu3_500_h200_fresh_0409_again"
source scripts/pipeline_common.sh
pipeline_setup

# Only train the missing WARM models
# (Skipping inverses to save queue time as requested)
GAPS=("warm_magnitude" "warm_fisher" "warm_momentum")
SPARSE_JIDS=()

for base in "${GAPS[@]}"; do
    m_path=$(ls "${MASK_OUT_BASE}/${RUN_ID}/${base}"*.pt 2>/dev/null | head -n 1)
    
    if [ -z "${m_path}" ]; then
        echo "ERROR: Could not find mask for ${base} in ${MASK_OUT_BASE}/${RUN_ID}/"
        exit 1
    fi

    echo "Submitting Sparse DPO for: $(basename "${m_path}")"
    jid=$(sbatch --parsable \
      --partition="${SPARSE_SLURM_PARTITION}" --gres="${SPARSE_SLURM_GRES}" --time=07:45:00 --mem=128G \
      --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}",PIPELINE_MASK_FILE="${m_path}",NUM_STEPS_DPO=500,DPO_LEARNING_RATE=5e-7,DPO_GRADIENT_CHECKPOINTING=1 \
      scripts/pipeline_sparse_one_mask.sh)
    SPARSE_JIDS+=("${jid}")
done

# Chain the final Eval job to the new sparse jobs
JOINS=$(IFS=:; echo "${SPARSE_JIDS[*]}")
echo "Chaining Eval job to: ${JOINS}"
sbatch --dependency=afterok:"${JOINS}" \
  --export=ALL,PIPELINE_RUN_ID="${RUN_ID}",RUN_ID="${RUN_ID}" \
  scripts/pipeline_stage_05_evals.sh

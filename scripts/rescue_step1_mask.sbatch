#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=mk_mom
#SBATCH --output=logs/rescue_mom_%j.out

set -e
export PIPELINE_RUN_ID="tulu3_500_h200_fresh_0409_again"
source scripts/pipeline_common.sh
pipeline_setup

# Build the missing momentum mask (now natively in bool format)
python src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "${TRAIN_OUT_BASE}/${RUN_ID}/deltas/meta_llama_llama_3_1_8b_instruct_tulu3" \
    --method momentum \
    --sparsity_percent 97.5 \
    --target_step 200 \
    --min_layer_keep_ratio 0.0025 \
    --output_file "${MASK_OUT_BASE}/${RUN_ID}/warm_momentum_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200.pt"

# Generate the inverse for the eval suite comparisons
python -c "from src.utils.mask_utils import invert_mask_file; invert_mask_file('${MASK_OUT_BASE}/${RUN_ID}/warm_momentum_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200.pt')"

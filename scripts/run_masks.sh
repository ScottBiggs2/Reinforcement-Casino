#!/bin/bash
# Ablation: DPO baseline and mask generation comparison
# Submit from anywhere: sbatch scripts/run_masks.sh
#SBATCH --job-name=dpo_masks
#SBATCH --output=logs/dpo_masks_%j.out
#SBATCH --error=logs/dpo_masks_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00 

# Resolve repo root (Slurm copies scripts to spool — use submit directory)
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

ENV_PATH="/scratch/biggs.s/conda_envs/rl_casino"
export PATH="$ENV_PATH/bin:$PATH"

# Set up PYTHONPATH
export PYTHONPATH=.

# Hyperparameters
MODEL="google/gemma-3-270m-it"
SPARSITY=97.5
TARGET_STEP=50 # Ensure this matches a saved step in DPO_train.py (e.g. 100, 150, 200, 250)
DATASET="qihoo360/Light-R1-DPOData"
LOG_DIR="delta_logs_google_gemma_3_270m_it"
MIN_LAYER_KEEP_RATIO="0.0025" # set to 0.0 for pure global masking


echo ""
echo "=================================================="
echo "1. Generating Warm-Start Masks"
echo "=================================================="

echo "-> Ground Truth Mask"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method magnitude \
  --sparsity_percent $SPARSITY \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard 

REF_MASK_GT="masks/warm_magnitude_google_gemma_3_270m_it_sparsity${SPARSITY}pct.pt"

echo "-> Magnitude Mask"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method magnitude \
  --sparsity_percent $SPARSITY \
  --target_step $TARGET_STEP \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard 

echo "-> Momentum Mask"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method momentum \
  --sparsity_percent $SPARSITY \
  --target_step $TARGET_STEP \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard 

echo "-> Fisher Mask (Warm)"
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$LOG_DIR" \
  --method fisher \
  --sparsity_percent $SPARSITY \
  --target_step $TARGET_STEP \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compute_jaccard 

echo ""
echo "=================================================="
echo "2. Generating Cold-Start Masks"
echo "=================================================="

echo "-> Fisher Mask (Cold, Improved)"
python src/cold_start/cold_mask_finder.py \
  --model_name "$MODEL" \
  --dataset_name "$DATASET" \
  --sparsity_percent $SPARSITY \
  --n_calibration_samples 256 \
  --mini_batch_size 4 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --reference_mask "$REF_MASK_GT"

echo "-> CAV Mask (Cold)"
python src/cold_start/cav_cold_mask_finder.py \
  --model_name "$MODEL" \
  --dataset_name "$DATASET" \
  --method cav \
  --sparsity_percent $SPARSITY \
  --subset_size 256 \
  --num_batches 16 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --reference_mask "$REF_MASK_GT"

echo ""
echo "=================================================="
echo "3. Comparing Against Random Baseline Masks"
echo "=================================================="
# We use the magnitude mask as the reference to extract parameter shapes

echo "-> vs. Ground Truth Mask"
REF_MASK="$REF_MASK_GT"
python src/warm_start/random_mask_baseline.py \
  --reference_mask "$REF_MASK" \
  --sparsity_percent $SPARSITY \
  --seed 42 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compare_to_reference \
  --output_file "masks/random_baseline_vs_ground_truth_sparsity${SPARSITY}pct.pt"

echo "-> vs. Magnitude Mask (Warm)"
REF_MASK="masks/warm_magnitude_google_gemma_3_270m_it_sparsity${SPARSITY}pct_step${TARGET_STEP}.pt"
python src/warm_start/random_mask_baseline.py \
  --reference_mask "$REF_MASK" \
  --sparsity_percent $SPARSITY \
  --seed 42 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compare_to_reference \
  --output_file "masks/random_baseline_vs_magnitude_sparsity${SPARSITY}pct.pt"

echo "-> vs. Momentum Mask (Warm)"
REF_MASK="masks/warm_momentum_w5_google_gemma_3_270m_it_sparsity${SPARSITY}pct_step${TARGET_STEP}.pt"
python src/warm_start/random_mask_baseline.py \
  --reference_mask "$REF_MASK" \
  --sparsity_percent $SPARSITY \
  --seed 42 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compare_to_reference \
  --output_file "masks/random_baseline_vs_momentum_sparsity${SPARSITY}pct.pt"

echo "-> vs. Fisher Mask (Warm)"
REF_MASK="masks/warm_fisher_google_gemma_3_270m_it_sparsity${SPARSITY}pct_step${TARGET_STEP}.pt"
python src/warm_start/random_mask_baseline.py \
  --reference_mask "$REF_MASK" \
  --sparsity_percent $SPARSITY \
  --seed 42 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compare_to_reference \
  --output_file "masks/random_baseline_vs_fisher_warm_sparsity${SPARSITY}pct.pt"

DATASET_SANITIZED="${DATASET//\//_}"

echo "-> vs. Fisher Mask (Cold)"
REF_MASK="masks/cold_fisher_google_gemma_3_270m_it_${DATASET_SANITIZED}_sparsity${SPARSITY}pct_n256.pt"
python src/warm_start/random_mask_baseline.py \
  --reference_mask "$REF_MASK" \
  --sparsity_percent $SPARSITY \
  --seed 42 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compare_to_reference \
  --output_file "masks/random_baseline_vs_fisher_cold_sparsity${SPARSITY}pct.pt"

echo "-> vs. CAV/SNIP Mask (Cold)"
REF_MASK="masks/cold_cav_google_gemma_3_270m_it_sparsity${SPARSITY}pct.pt"
python src/warm_start/random_mask_baseline.py \
  --reference_mask "$REF_MASK" \
  --sparsity_percent $SPARSITY \
  --seed 42 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --compare_to_reference \
  --output_file "masks/random_baseline_vs_cav_cold_sparsity${SPARSITY}pct.pt"


echo ""
echo "=================================================="
echo "All done!"
echo "=================================================="

#!/bin/bash
# Verification: Run 10 steps of GRPO training (Dense and Sparse) on datasets.
# Submit from project root: sbatch verify_grpo_training.sh
#SBATCH --job-name=verify_grpo
#SBATCH --output=logs/verify_grpo_%j.out
#SBATCH --error=logs/verify_grpo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00

# Job runs in the directory you submitted from
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
conda activate /scratch/biggs.s/conda_envs/rl_casino

export PYTHONPATH=.
echo "Installing/verifying training requirements..."
pip install -r requirements.txt -q

MODEL="google/gemma-3-270m-it"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
NUM_STEPS=10
SUBSET=64
VERIFY_OUT_DIR="/scratch/biggs.s/rl_casino_verify_outputs"
VERIFY_CACHE_DIR="/scratch/biggs.s/hf_cache_verify"

# Default mask to verify the sparse script's parameter injection
DEFAULT_MASK="masks/top_10.0pct_momentum_w25_step25.pt"

DATASETS=("math-220k")
PASS_COUNT=0
FAIL_COUNT=0

echo ""
echo "============================================================"
echo "Verification: ${#DATASETS[@]} datasets × ${NUM_STEPS} steps each (Dense + Sparse)"
echo "Model: ${MODEL}, Subset: ${SUBSET}"
echo "============================================================"
echo ""

for DS in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "Testing DENSE GRPO on dataset: ${DS}"
    echo "============================================================"

    python src/full_training/GRPO_train.py \
        --model_name "$MODEL" \
        --dataset "$DS" \
        --num_steps "$NUM_STEPS" \
        --subset_size "$SUBSET" \
        --output_base_dir "$VERIFY_OUT_DIR" \
        --dataset_cache_dir "$VERIFY_CACHE_DIR" \
        --num_generations 4 \
        --generation_batch_size 4 \
        --use_wandb \
        --run_name "verify_grpo_dense_${DS}_$(date +%Y%m%d_%H%M%S)" 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ DENSE ${DS}: PASSED"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ DENSE ${DS}: FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    echo "============================================================"
    echo "Building Warm Start Magnitude Mask for testing: "
    echo "============================================================"
    SANITIZED_MODEL=$(echo "$MODEL" | tr '/-' '__' | tr '[:upper:]' '[:lower:]')
    SANITIZED_DS=$(echo "$DS" | tr '-' '_')
    DELTA_DIR="${VERIFY_OUT_DIR}/deltas/${SANITIZED_MODEL}_${SANITIZED_DS}_grpo_dense"
    
    GENERATED_MASK="masks/verify_${SANITIZED_MODEL}_${SANITIZED_DS}_step${NUM_STEPS}.pt"
    
    python src/warm_start/even_better_mask_finder.py \
        --delta_log_dir "$DELTA_DIR" \
        --method magnitude \
        --sparsity_percent 97.5 \
        --target_step "$NUM_STEPS" \
        --mlp_only \
        --output_file "$GENERATED_MASK"


    echo "============================================================"
    echo "Testing SPARSE GRPO on dataset: ${DS}"
    echo "============================================================"

    python src/full_training/sparse_grpo_bsr.py \
        --model_name "$MODEL" \
        --dataset "$DS" \
        --n_steps "$NUM_STEPS" \
        --subset_size "$SUBSET" \
        --optimizer sparse_adamw \
        --mask "$GENERATED_MASK" \
        --output_base_dir "$VERIFY_OUT_DIR" \
        --dataset_cache_dir "$VERIFY_CACHE_DIR" \
        --num_generations 4 \
        --generation_batch_size 4 \
        --use_wandb \
        --run_name "verify_grpo_sparse_${DS}_$(date +%Y%m%d_%H%M%S)" 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ SPARSE ${DS}: PASSED"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ SPARSE ${DS}: FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "============================================================"
echo "Verification Summary"
echo "============================================================"
TOTAL_TESTS=$((${#DATASETS[@]} * 2))
echo "Passed: ${PASS_COUNT} / ${TOTAL_TESTS}"
echo "Failed: ${FAIL_COUNT} / ${TOTAL_TESTS}"
echo ""

# Check that output directories were created correctly on scratch
echo "Checking output directories in ${VERIFY_OUT_DIR} for DENSE delta logs:"
for DS in "${DATASETS[@]}"; do
    SANITIZED=$(echo "$DS" | tr '-' '_')
    DIR_PATTERN="${VERIFY_OUT_DIR}/deltas/*_${SANITIZED}_grpo_dense*"
    if ls -d ${DIR_PATTERN} 2>/dev/null; then
        echo "  ✓ Found: ${DIR_PATTERN}"
    else
        echo "  ✗ Not found: ${DIR_PATTERN}"
    fi
done

echo ""
echo "Job finished at: $(date)"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "VERIFICATION FAILED: ${FAIL_COUNT} test(s) had errors"
    exit 1
else
    echo "ALL GRPO SCRIPTS VERIFIED SUCCESSFULLY"
    exit 0
fi

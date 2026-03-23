#!/bin/bash
# Verification: Run 10 steps of DPO training on each dataset to confirm everything works.
# Submit from project root: sbatch verify_training.sh
#SBATCH --job-name=verify_dpo_datasets
#SBATCH --output=logs/verify_training_%j.out
#SBATCH --error=logs/verify_training_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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

MODEL="google/gemma-3-270m-it"
NUM_STEPS=10
SUBSET=64
VERIFY_OUT_DIR="/scratch/biggs.s/rl_casino_verify_outputs"
VERIFY_CACHE_DIR="/scratch/biggs.s/hf_cache_verify"

DATASETS=("light-r1" "tulu3" "math-step-dpo" "codepref")
PASS_COUNT=0
FAIL_COUNT=0

echo ""
echo "============================================================"
echo "Verification: ${#DATASETS[@]} datasets × ${NUM_STEPS} steps each"
echo "Model: ${MODEL}, Subset: ${SUBSET}"
echo "============================================================"
echo ""

for DS in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "Testing dataset: ${DS}"
    echo "============================================================"

    python src/full_training/DPO_train.py \
        --model_name "$MODEL" \
        --dataset "$DS" \
        --num_steps "$NUM_STEPS" \
        --subset_size "$SUBSET" \
        --output_base_dir "$VERIFY_OUT_DIR" \
        --dataset_cache_dir "$VERIFY_CACHE_DIR" \
        --use_wandb \
        --run_name "verify_${DS}_$(date +%Y%m%d_%H%M%S)" 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ ${DS}: PASSED"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ ${DS}: FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "============================================================"
echo "Verification Summary"
echo "============================================================"
echo "Passed: ${PASS_COUNT} / ${#DATASETS[@]}"
echo "Failed: ${FAIL_COUNT} / ${#DATASETS[@]}"
echo ""

# Check that output directories were created correctly on scratch
echo "Checking output directories in ${VERIFY_OUT_DIR}:"
for DS in "${DATASETS[@]}"; do
    SANITIZED=$(echo "$DS" | tr '-' '_')
    # According to new path logic in DPO_train.py:
    # DELTA_LOG_DIR = os.path.join(BASE_DIR, "deltas", f"{MODEL_NAME_SANITIZED}_{DATASET_SANITIZED}")
    # So we search for folders under $VERIFY_OUT_DIR/deltas/
    DIR_PATTERN="${VERIFY_OUT_DIR}/deltas/*_${SANITIZED}"
    if ls -d ${DIR_PATTERN} 2>/dev/null; then
        echo "  ✓ Found: ${DIR_PATTERN}"
        # Check for metadata and delta files
        for D in ${DIR_PATTERN}; do
            if [ -f "${D}/run_metadata.json" ]; then
                echo "    ✓ run_metadata.json present"
            else
                echo "    ✗ run_metadata.json MISSING"
            fi
            if [ -f "${D}/stats_step_10.json" ]; then
                echo "    ✓ stats_step_10.json present"
            else
                echo "    ✗ stats_step_10.json MISSING"
            fi
            if [ -f "${D}/deltas_step_10.pt" ]; then
                echo "    ✓ deltas_step_10.pt present"
            else
                echo "    ✗ deltas_step_10.pt MISSING"
            fi
        done
    else
        echo "  ✗ Not found: ${DIR_PATTERN}"
    fi
done

echo ""
echo "Job finished at: $(date)"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "VERIFICATION FAILED: ${FAIL_COUNT} dataset(s) had errors"
    exit 1
else
    echo "ALL DATASETS VERIFIED SUCCESSFULLY"
    exit 0
fi

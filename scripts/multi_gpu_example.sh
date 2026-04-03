#!/bin/bash
#SBATCH --job-name=qwen3-scaling
#SBATCH --reservation=biggs.s_test
#SBATCH --partition=reservation
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=logs/scaling-%j.out
#SBATCH --err=logs/scaling-%j.err

# === CONFIGURATION ===
MODEL_NAME="Qwen/Qwen3-0.6B"
DATA_DIR="/scratch/$USER/data/sft/qwen3-0.6b/tulu-3"
COMMON_OUTPUT_DIR="/scratch/$USER/outputs/scaling_tests"
WANDB_PROJECT="qwen3-scaling-test"
MAX_STEPS=500
LOGGING_STEPS=1

# === ENVIRONMENT SETUP ===
export HF_HOME="/scratch/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
export PYTHONPATH="$(pwd)/dllm:$(pwd):$PYTHONPATH"
export WANDB_PROJECT=$WANDB_PROJECT

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$COMMON_OUTPUT_DIR"

# Activate environment
module load cuda/12.3.0
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate /scratch/$USER/project_envs/qwen3_dllm

echo "Starting Scaling Efficiency Test Suite"
echo "Reservation: biggs.s_test"
echo "Node: $SLURMD_NODENAME"
echo "Project: $WANDB_PROJECT"

# Loop through GPU configurations
# for NUM_GPUS in 1, 2 4; do 
for NUM_GPUS in 2 4; do 
    RUN_NAME="qwen3-scaling-${NUM_GPUS}gpu"
    OUTPUT_DIR="${COMMON_OUTPUT_DIR}/${RUN_NAME}"
    
    echo "=================================================="
    echo "Starting test with ${NUM_GPUS} GPUs..."
    echo "Run Name: ${RUN_NAME}"
    echo "=================================================="
    
    # Strong Scaling: Keep global batch size constant at 32
    # Global BS = Per_Device (8) * Num_GPUs * Grad_Acc
    # 32 = 8 * NUM_GPUS * GRAD_ACC
    # GRAD_ACC = 4 / NUM_GPUS
    
    GRAD_ACC=$((4 / NUM_GPUS))
    
    echo "Global Batch Size: 32"
    echo "Gradient Accumulation Steps: $GRAD_ACC"

    # Run training
    # We explicitly set --num_processes to control the number of GPUs used by accelerate
    accelerate launch \
        --config_file dllm/scripts/accelerate_configs/zero2.yaml \
        --num_processes $NUM_GPUS \
        scripts/train_qwen3_mdlm.py \
        --model_name_or_path "$MODEL_NAME" \
        --dataset_args "$DATA_DIR" \
        --load_preprocessed_data True \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --report_to wandb \
        --logging_steps $LOGGING_STEPS \
        --max_steps $MAX_STEPS \
        --save_strategy "no" \
        --eval_strategy "no" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps $GRAD_ACC \
        --learning_rate 2e-5 \
        --warmup_ratio 0.1 \
        --overwrite_output_dir
        
    echo "Completed test with ${NUM_GPUS} GPUs."
    echo ""
done

echo "All scaling tests completed."
export HF_TOKEN="hf_YourTokenHere"

sbatch scripts/run_evals_slurm.sh \
  --model_path "$HOME/rl_casino_train/tulu3_500_h200_fresh_0409_again/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500" \
  --trust_remote_code \
  --output_dir "results/eval_tulu3_500_h200_fresh_0409_again_ckpt500"

  Now we can work with this for several evals here: 

  /scratch/biggs.s/rl_casino_sparse_train/tulu3_500_h200_fresh_0409_again/checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct/fullpipe_sparse_checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct_tulu3_500_h200_fresh_0409_again/final_model

  sbatch scripts/run_evals_slurm.sh \
  --model_path /scratch/biggs.s/rl_casino_sparse_train/tulu3_500_h200_fresh_0409_again/checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct/fullpipe_sparse_checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct_tulu3_500_h200_fresh_0409_again/final_model \
  --trust_remote_code \
  --output_dir "results/eval_tulu3_500_h200_fresh_0409_again_ckpt500"


Sparse speed ablation: 

(base) [biggs.s@explorer-02 rl_casino]$ export H200_BSR_OUT="${SCRATCH_USER_ROOT}/rl_casino_h200_bsr/run_${SLURM_JOB_ID:-manual}"
(base) [biggs.s@explorer-02 rl_casino]$ export SCRATCH_USER_ROOT="/scratch/${USER}"
(base) [biggs.s@explorer-02 rl_casino]$ export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export H200_BSR_STEPS_PER_PHASE=100
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_MAX_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=1024
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
export DPO_GRADIENT_ACCUMULATION_STEPS=64
export HF_DATASETS_CACHE="${SCRATCH_USER_ROOT}/hf_cache/datasets"
(base) [biggs.s@explorer-02 rl_casino]$ unset NUM_STEPS_DPO 2>/dev/null || true
(base) [biggs.s@explorer-02 rl_casino]$ export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_CONSOLE=off
(base) [biggs.s@explorer-02 rl_casino]$ sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
Submitted batch job XXXXXX
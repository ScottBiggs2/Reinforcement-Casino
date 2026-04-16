export HF_TOKEN="hf_YourTokenHere"

sbatch scripts/run_evals_slurm.sh \
  --model_path "$HOME/rl_casino_train/tulu3_500_h200_fresh_0409_again/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500" \
  --trust_remote_code \
  --output_dir "results/eval_tulu3_500_h200_fresh_0409_again_ckpt500"

  Now we can work with this for several evals here: 

  
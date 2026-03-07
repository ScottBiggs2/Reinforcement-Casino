#!/bin/bash
# Slurm job script to run the full evaluation suite
# Usage: sbatch run_evals_slurm.sh --model_path <HUGGINGFACE_ID_OR_PATH>

#SBATCH --job-name=llm_eval_suite
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Job runs in the directory you submitted from
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results

echo "Evaluation started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Conda environment activation
# if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
# elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#   source "$HOME/anaconda3/etc/profile.d/conda.sh"
# fi

# Activate environment (adjust name if different)
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
conda activate /scratch/biggs.s/conda_envs/rl_casino

# Install/Verify lm-eval
bash install_lm_eval.sh

echo "================================"
echo "Running All Benchmarks"
echo "================================"

# Pass all arguments from sbatch to the python script
# Example: sbatch run_evals_slurm.sh --model_path meta-llama/Llama-3.1-8B-Instruct --limit 100
python src/evaluation/run_all_benchmarks.py \
  --output_dir "results/eval_${SLURM_JOB_ID}" \
  --verbose \
  "$@"

echo "================================"
echo "Evaluation finished at: $(date)"
echo "Results saved to: results/eval_${SLURM_JOB_ID}"
echo "================================"

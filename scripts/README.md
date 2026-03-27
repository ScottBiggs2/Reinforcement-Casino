# Scripts Directory

Operational shell scripts live here so the repository root stays focused on source code.

## How to run

- **Submit from the repository root** (required for Slurm):
  - `cd /path/to/rl_casino && sbatch scripts/run_evals_slurm.sh --model_path "meta-llama/Llama-3.1-8B-Instruct"`
  - `sbatch scripts/verify_coding.sh --model_path "google/gemma-3-270m-it"`
  - `sbatch scripts/verify_training.sh --model_path "google/gemma-3-270m-it"`
  - `sbatch scripts/verify_grpo_training.sh --model_path "google/gemma-3-270m-it"`
- Slurm **copies** the batch script to `/var/spool/slurmd/...`, so scripts resolve the repo via **`SLURM_SUBMIT_DIR`** (the directory you ran `sbatch` from), not via `BASH_SOURCE`. Running `sbatch` from the wrong directory makes `cd` land in `/var/spool/slurmd` and breaks `mkdir logs`, Python paths, etc.

## Quick index

- `run_evals_slurm.sh` - full benchmark eval runner.
- `verify_coding.sh` - short coding benchmark verification.
- `verify_training.sh` - short DPO training verification across datasets.
- `verify_grpo_training.sh` - short GRPO dense+sparse verification.
- `run_masks.sh` - warm/cold/random mask comparison workflow.
- `run_dpo_and_masks.sh` - DPO + mask generation workflow.
- `run_mask_diagnostics.sh` - attention masking diagnostic.
- `install_lm_eval.sh` - installs lm-eval and related dependencies.
- `run_ablation_*.sh` - targeted ablation workflows.

## Notes

- If your HPC environment requires different paths, update:
  - `ENV_PATH` (Conda env path)
  - `#SBATCH` resource directives
- Keep logs under `logs/` and outputs under `results/` or scratch directories.

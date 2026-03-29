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
- `run_full_pipeline.sh` - **one Slurm job** for the entire train → masks → comparisons → sparse → eval flow (default **8h** wall; use the chain for longer work).
- `submit_pipeline_chain.sh` + `pipeline_stage_01_dense.sh` … `pipeline_stage_05_evals.sh` - **chained jobs** (`afterok`) for sites with a short per-job wall limit (e.g. 8h). Shared logic lives in `pipeline_common.sh`.
- `pipeline_sparse_one_mask.sh` - one sparse DPO run per mask; **stage 4** submits one Slurm job per `*.pt` under the run’s mask dir (parallel), then queues evals when all finish (`SPARSE_SLURM_TIME` default **8h** per mask).

## Full RL Casino pipeline (Tulu3 / Llama 3.1 8B IT)

End-to-end flow: **dense DPO → warm/cold masks → mask comparisons → parallel sparse DPO (one Slurm job per mask) → benchmark eval `sbatch` fan-out**.

| Mode | When to use | Command |
|------|-------------|---------|
| **Chained jobs** | Per-job **wall limit** (e.g. 8h). Stages chain with `afterok`; sparse runs **in parallel** on separate GPUs. | **Login-node sample** below. |
| **Single job** | One allocation **≤8h** (fits typical cluster max; tight for full pipeline). | `sbatch scripts/run_full_pipeline.sh` |

Defaults and scratch paths are in [`pipeline_common.sh`](pipeline_common.sh). Override with **environment variables** before launching (same names as in that file: `MODEL`, `DPO_DATASETS`, `NUM_STEPS_DPO`, `TARGET_STEP_DPO`, eval knobs, etc.).

### Chained pipeline — copy/paste (login node)

Run **from the repository root** (the directory that contains `src/` and `scripts/`).

```bash
# --- 0) Go to repo root ---
cd /path/to/rl_casino

# --- 1) Gated models (Llama 3.1) ---
export HF_TOKEN="hf_xxxxxxxx"   # required if the hub gates the model

# --- 2) Optional: fixed run id (default: submit_pipeline_chain.sh picks a timestamp id) ---
# export PIPELINE_RUN_ID="my_experiment_20260329"

# --- 3) Optional overrides (uncomment as needed) ---
# export PIPELINE_SPARSE_EVAL_DEPENDENCY=afterany   # eval stage runs after all sparse jobs *finish* (even if some failed)
# export SPARSE_SLURM_TIME=08:00:00                 # wall time per parallel sparse job (default 8h; cluster max)
# export RUN_MASK_CKA=1                             # enable mask CKA in comparisons (GPU-heavy)
# export EVAL_LIMIT=100                             # cap benchmark size; omit or empty for full runs

# --- 4) Launch the chain (stage 1 = dense DPO; later stages auto-submit via Slurm dependencies) ---
bash scripts/submit_pipeline_chain.sh
```

The script prints **stage 1**’s Slurm job id and `PIPELINE_RUN_ID`. Monitor:

```bash
squeue -u "$USER"
# Stage 1 log (replace JOBID with the id printed above):
tail -f logs/pipeline_JOBID_p1_dense.out
```

**Where outputs go** (see `TRAIN_OUT_BASE`, `MASK_OUT_BASE`, etc. in `pipeline_common.sh` — often under `/scratch/.../rl_casino_*`):

- Dense training: `TRAIN_OUT_BASE/${PIPELINE_RUN_ID}/`
- Masks: `MASK_OUT_BASE/${PIPELINE_RUN_ID}/`
- Comparisons: `MASK_OUT_BASE/${PIPELINE_RUN_ID}/comparisons/`
- Sparse runs: `SPARSE_OUT_BASE/${PIPELINE_RUN_ID}/<mask_stem>/`
- Parallel sparse **per-job logs**: `logs/sparse_${PIPELINE_RUN_ID}_<mask_stem>_*.out`
- Eval harness: `EVAL_OUT_BASE/${PIPELINE_RUN_ID}/...` (plus each eval `sbatch` log under `logs/`)

**Requirements:** nested `sbatch` (submitting the next stage from inside a running job) must be allowed. Edit `#SBATCH` lines (`partition`, `gres`, `mem`, `time`) in `pipeline_stage_*.sh`, `pipeline_sparse_one_mask.sh`, and `run_full_pipeline.sh` to match your cluster.

**Slurm submit directory:** Always run `sbatch` or `bash scripts/submit_pipeline_chain.sh` from the **repository root** (the directory you `cd` into before submitting). Slurm sets `SLURM_SUBMIT_DIR` to that path; the pipeline uses it to find `scripts/pipeline_common.sh`. If you submit from elsewhere, sourcing `pipeline_common.sh` fails with “No such file” under `/var/spool/slurmd/...`.

### Single long job (no parallel sparse workers)

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"   # if needed
sbatch scripts/run_full_pipeline.sh
```

Sparse DPO runs **sequentially** inside this job. Use the chained pipeline if you want parallel sparse GPUs.

## Notes

- If your HPC environment requires different paths, update:
  - `TRAIN_ENV` / `EVAL_ENV` in [`pipeline_common.sh`](pipeline_common.sh)
  - `#SBATCH` resource directives in each `pipeline_stage_*.sh` and `run_full_pipeline.sh`
- Keep logs under `logs/` and outputs under `results/` or scratch directories.

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
- `run_full_pipeline.sh` - **one Slurm job** for the entire train → masks → comparisons → sparse → eval flow (wall **≤8h**; full pipelines usually need the **chain** instead).
- `submit_pipeline_chain.sh` + `pipeline_stage_01_dense.sh` … `pipeline_stage_05_evals.sh` - **chained jobs** (`afterok`) so each stage respects a typical **8h max** wall. GPU-heavy stages default to **7h45** Slurm time with **7h30** soft `timeout`s; stage **4** (sparse launcher) and **5** (eval fan-out) use the **cpu** partition with small mem. Shared logic lives in `pipeline_common.sh`.
- `pipeline_sparse_one_mask.sh` - one sparse DPO run per mask; **stage 4** submits one Slurm job per `*.pt` under the run’s mask dir (parallel), then queues evals when all finish (`SPARSE_SLURM_TIME` default **7h45** per mask; override `SPARSE_SLURM_MEM` if needed).
- `multigpu_pipeline/` - **parallel multi-GPU entrypoints** (keeps the single-GPU pipeline unchanged). Currently provides a multi-GPU dense DPO stage 1 launcher.

## Full RL Casino pipeline (Tulu3 / Llama 3.1 8B IT)

End-to-end flow: **dense DPO → warm/cold masks → mask comparisons → parallel sparse DPO (one Slurm job per mask) → benchmark eval `sbatch` fan-out**.

| Mode | When to use | Command |
|------|-------------|---------|
| **Chained jobs** | Per-job **wall limit** (e.g. 8h). Stages chain with `afterok`; sparse runs **in parallel** on separate GPUs. | **Login-node sample** below. |
| **Single job** | One allocation **≤8h** — rarely enough wall for a full multi-stage pipeline end-to-end. | `sbatch scripts/run_full_pipeline.sh` |

Defaults and scratch paths are in [`pipeline_common.sh`](pipeline_common.sh). Override with **environment variables** before launching (same names as in that file: `MODEL`, `DPO_DATASETS`, `NUM_STEPS_DPO`, `TARGET_STEP_DPO`, eval knobs, etc.).

**Slurm / resources:** GPU stages use `#SBATCH --time=07:45:00` (under a typical **8h** cap). `TRAIN_TIMEOUT_PER_DATASET`, `MASK_TIMEOUT`, and `SPARSE_TIMEOUT_PER_MASK` default to **7h30m** so processes get SIGTERM before Slurm. CPU-only stage **3a** (Jaccard / CSV) uses `PIPELINE_CPU_COMPARISON_TIME` (default **6h**), `PIPELINE_CPU_COMPARISON_MEM` (**64G**), `PIPELINE_CPU_COMPARISON_CPUS` (**4**). If your site has no `cpu` partition, edit `#SBATCH --partition` in `pipeline_stage_04_sparse.sh` and `pipeline_stage_05_evals.sh` (and the `sbatch` lines in `pipeline_stage_02_masks.sh` / `resume_pipeline_from_stage.sh` for stage 3a).

### Single-GPU dense DPO — Tulu3 paper-style hyperparams (`SEQ` 1024)

The paper’s global batch 128 used **8 GPUs** (per-device 1 × grad-accum 16 × 8). On **one GPU**, match global batch 128 with **per-device 1 × grad-accum 128** (or any product \(=128\)).

Optional env vars are read by `run_dense_dpo()` in [`pipeline_common.sh`](pipeline_common.sh): `DPO_PER_DEVICE_TRAIN_BATCH_SIZE`, `DPO_GRADIENT_ACCUMULATION_STEPS`, `DPO_LEARNING_RATE`, `DPO_WARMUP_RATIO`, `DPO_WEIGHT_DECAY`, `DPO_MAX_LENGTH`, `DPO_MAX_PROMPT_LENGTH`, `DPO_BETA`. Gradient checkpointing defaults **on**; set `DPO_GRADIENT_CHECKPOINTING=0` to disable.

**Chained full pipeline** (stage 1 uses these DPO settings):

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"

export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export DPO_DATASETS="tulu3"
export NUM_STEPS_DPO=2000

# Paper-style (Tulu3 DPO); sequence length 1024 for faster steps than 2048
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=1
export DPO_GRADIENT_ACCUMULATION_STEPS=128
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_WEIGHT_DECAY=0.0
export DPO_MAX_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=512

export DELTA_LOG_INTERVAL=50
export DELTA_LOG_END_STEP=200
export TARGET_STEP_DPO=200

# Dense DPO must finish within the Slurm wall (7h45 request, 7h30 soft timeout by default).
# If you hit the limit, reduce steps / improve throughput — do not request >8h if the cluster forbids it.

bash scripts/submit_pipeline_chain.sh
```

**Stage 1 only** (same env, then):

```bash
export PIPELINE_RUN_ID="singlegpu_tulu3_paper_$(date +%Y%m%d_%H%M%S)"
export RUN_ID="$PIPELINE_RUN_ID"
sbatch --export=ALL,PIPELINE_RUN_ID,RUN_ID scripts/pipeline_stage_01_dense.sh
```

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
# export SPARSE_SLURM_TIME=07:45:00                 # wall time per parallel sparse GPU job (default; ≤8h cluster max)
# export SPARSE_SLURM_MEM=96G                       # optional: lower host RAM if your cluster charges by mem
# export PIPELINE_CPU_COMPARISON_TIME=08:00:00      # optional: extend stage 3a wall if Jaccard/CSV needs it (default 6h)
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

## Multi-GPU dense DPO (canonical: `tulu3`, Llama 3.1 8B)

If you have access to a multi-GPU reservation/partition, you can run **dense DPO stage 1** with multiple GPUs using the parallel scripts in `scripts/multigpu_pipeline/`.

This is intentionally scoped to **dense DPO only** (mask/sparse/evals remain on the existing pipeline unless/until you add multi-GPU versions). This avoids overcomplicating requirements and lets you debug multi-process training in isolation.

### Slurm batch (copy/paste)

From the repo root on the login node:

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"   # required for gated Llama downloads

# Canonical verification case
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export DPO_DATASET_KEY="tulu3"

# Multi-GPU controls (must match the script's #SBATCH --gres)
# Use "auto" to match the number of visible GPUs (recommended).
# Or set an integer that matches your Slurm --gres allocation.
export MULTIGPU_NGPUS=auto

# Paper-parity defaults (arXiv:2505.11711 Appendix B); override as needed.
export SEQ_LEN=2048
export PER_DEVICE_BS=1
export GRAD_ACCUM=16
export LR_PEAK=5e-7
export WARMUP_RATIO=0.1
export WEIGHT_DECAY=0.0
export NUM_EPOCHS=1

# Optional: faster debug run
# export SUBSET_DPO=256
# export NUM_EPOCHS=0.05
# export DELTA_LOG_END_STEP=50

sbatch scripts/multigpu_pipeline/pipeline_stage_01_dense_dpo_multigpu.sh
```

### Full pipeline (multi-GPU stage 1 → existing stages 2–4)

To relaunch the **full artifact pipeline** while using the new **multi-GPU-compatible dense DPO stage 1**, do:

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"

# Canonical verification case
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export DPO_DATASET_KEY="tulu3"

# 2000-step trial run (canonical hyperparams + warm-start artifacts)
export NUM_STEPS_DPO=2000
export SEQ_LEN=2048
export PER_DEVICE_BS=1
export GRAD_ACCUM=16
export LR_PEAK=5e-7
export WARMUP_RATIO=0.1
export WEIGHT_DECAY=0.0

# Multi-GPU controls (must match the script's #SBATCH --gres)
export MULTIGPU_NGPUS=auto

# Keep warm-start artifact schedule compatible with the single-GPU pipeline defaults
export DELTA_LOG_INTERVAL=50
export DELTA_LOG_END_STEP=200
export TARGET_STEP_DPO=200

# Choose a run id so stage 2+ can find stage 1 outputs
export PIPELINE_RUN_ID="mgpu_tulu3_2000steps_$(date +%Y%m%d_%H%M%S)"
export RUN_ID="$PIPELINE_RUN_ID"

# Submit stage 1 (multi-GPU dense DPO)
J1=$(sbatch --parsable --export=ALL,PIPELINE_RUN_ID,RUN_ID scripts/multigpu_pipeline/pipeline_stage_01_dense_dpo_multigpu.sh)
echo "Stage 1 (multigpu dense DPO) job id: $J1  RUN_ID=$RUN_ID"

# Chain stage 2 (masks) after stage 1 completes.
J2=$(sbatch --parsable --dependency=afterok:"$J1" --export=ALL,PIPELINE_RUN_ID="$RUN_ID",RUN_ID="$RUN_ID" scripts/pipeline_stage_02_masks.sh)
echo "Stage 2 (masks) job id: $J2"

# Stage 2 will chain stage 3 CPU comparisons → stage 4 sparse → stage 5 eval submissions via existing scripts.
```

Success/progress checks:

```bash
squeue -u "$USER"

# Dense DPO logs (stage 1):
tail -f logs/pipeline_${J1}_p1_dense_mgpu.out
tail -f logs/full_pipeline_dpo_multigpu_tulu3_${RUN_ID}.log

# After stage 1, verify warm-start artifacts exist:
ls -lh /scratch/biggs.s/rl_casino_train/${RUN_ID}/deltas/*/base_state.pt
ls -lh /scratch/biggs.s/rl_casino_train/${RUN_ID}/deltas/*/deltas_step_*.pt | head
```

### Interactive allocation (debug)

If you prefer an interactive shell under a reservation, follow your site guidance. Explorer documents the reservation pattern here:
- https://rc-docs.northeastern.edu/en/explorer-main/gpus/multigpu-partition-access.html

Once you have an interactive shell with N GPUs, you can run the same command line as the sbatch script uses, e.g.:

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"
export MODEL="meta-llama/Llama-3.1-8B-Instruct"

export MULTIGPU_NGPUS=4
/scratch/biggs.s/conda_envs/rl_casino/bin/torchrun --standalone --nproc_per_node="$MULTIGPU_NGPUS" \
  src/full_training/DPO_train.py \
    --model_name "$MODEL" \
    --dataset "tulu3" \
    --num_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_length 2048 \
    --max_prompt_length 2048 \
    --delta_log_interval 50 \
    --delta_log_end_step 200 \
    --output_base_dir "/scratch/biggs.s/rl_casino_train/manual_${SLURM_JOB_ID:-local}" \
    --dataset_cache_dir "/scratch/biggs.s/hf_cache/datasets" \
    --use_wandb \
    --run_name "manual_multigpu_dpo_${SLURM_JOB_ID:-local}"
```

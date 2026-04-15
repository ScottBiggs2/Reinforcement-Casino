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
- `invert_mask.py` - CLI for complement masks (`1−M`); pipeline writes `<stem>_inverse.pt` per primary mask when `PIPELINE_GENERATE_INVERSE_MASKS=1` (default).
- `verify_inverse_masks.sh` - check `*_inverse.pt` counts vs primaries under `MASK_OUT_BASE/<RUN_ID>/` and grep complement lines in `logs/full_pipeline_masks_<RUN_ID>.log`.
- `wipe_pipeline_artifacts.sh` - **destructive** reset: optional `scancel -u $USER`, remove `--run-id` subtrees or `--full-scratch` trees, optional `--repo-logs`, optional `--include-hf-cache`. Requires `--yes`. See script header.
- `run_dpo_and_masks.sh` - DPO + mask generation workflow.
- `run_mask_diagnostics.sh` - attention masking diagnostic.
- `submit_gen_grpo_masks_gpu.slurm` - GPU-safe Slurm launcher for GRPO mask generation + plots inside Apptainer/Singularity (`--nv` + CUDA preflight).
- `install_lm_eval.sh` - installs lm-eval and related dependencies.
- `run_ablation_*.sh` - targeted ablation workflows.
- `run_full_pipeline.sh` - **one Slurm job** for the entire train → masks → comparisons → sparse → eval flow (wall **≤8h**; full pipelines usually need the **chain** instead).
- `submit_pipeline_chain.sh` + `pipeline_stage_01_dense.sh` … `pipeline_stage_05_evals.sh` - **chained jobs** (`afterok`) so each stage respects a typical **8h max** wall. GPU-heavy stages default to **7h45** Slurm time with **7h30** soft `timeout`s. **Stage 2** is **split**: [`pipeline_stage_02a_masks_warm.sh`](pipeline_stage_02a_masks_warm.sh) (**GPU**, warm delta masks — **delta streaming accumulation uses CUDA** when `RL_CASINO_WARM_MASK_SCORE_DEVICE=cuda` (default in `pipeline_setup`). A **GPU preflight** matmul runs at job start so allocations are not “idle.” For models with huge parameter counts, `mask_utils.create_mask_from_scores_gpu_efficient` may print that the **chunked global selector** moved scores to **CPU** for the final top-k; most wall time is still typically GPU-bound during delta passes. Set `RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu` only if you must avoid GPU or hit OOM. **→** [`pipeline_stage_02b_masks_cold.sh`](pipeline_stage_02b_masks_cold.sh) (GPU, Fisher+CAV) → [`pipeline_stage_02c_masks_post.sh`](pipeline_stage_02c_masks_post.sh) (CPU, random+inverses). [`pipeline_stage_02_masks.sh`](pipeline_stage_02_masks.sh) is a **short launcher** (~30m) that only `sbatch`-submits 02a (inherits 02a’s GPU `#SBATCH`; do not override `--partition` to a CPU queue unless you also set `RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu`). **Stage 4** (sparse launcher) is submitted at the **start** of stage 3 so sparse GPU jobs overlap mask comparisons; stage **4**’s launcher and **5** (eval fan-out) use the **cpu** partition with small mem. Shared logic lives in `pipeline_common.sh`.
- `pipeline_sparse_one_mask.sh` - one sparse DPO run per mask; **stage 4** submits one Slurm job per `*.pt` under the run’s mask dir (parallel), then queues evals when all finish (`SPARSE_SLURM_TIME` default **7h45** per mask; override `SPARSE_SLURM_MEM` if needed).
- `multigpu_pipeline/` - **parallel multi-GPU entrypoints** (keeps the single-GPU pipeline unchanged). Currently provides a multi-GPU dense DPO stage 1 launcher.
- `h200_sparse_dpo_bsr_benchmark.sh` - **single H200 GPU** Slurm job: multi-phase BSR DPO throughput benchmark (dense AdamW + random-mask sparse phases). See [H200 BSR DPO benchmark](#h200-bsr-dpo-benchmark-throughput-and-random-masks) below.
- `plot_h200_bsr_benchmark.py` - **plots** `benchmark_training_log.csv` from that benchmark (throughput, loss, summary bars). Run locally or on a login node with pandas/matplotlib.

## Full RL Casino pipeline (Tulu3 / Llama 3.1 8B IT)

End-to-end flow: **dense DPO → warm/cold masks + random baseline + complement (`*_inverse.pt`) masks → stage 3 starts parallel sparse DPO (`sbatch` per mask `.pt`) while running mask comparisons (structured + extended pairs, random vs structured, complement sanity + cross pairs) → benchmark eval `sbatch` fan-out** after sparse jobs finish.

| Mode | When to use | Command |
|------|-------------|---------|
| **Chained jobs** | Per-job **wall limit** (e.g. 8h). Stages chain with `afterok`; sparse runs **in parallel** on separate GPUs. | **Login-node sample** below. |
| **Single job** | One allocation **≤8h** — rarely enough wall for a full multi-stage pipeline end-to-end. | `sbatch scripts/run_full_pipeline.sh` |

Defaults and scratch paths are in [`pipeline_common.sh`](pipeline_common.sh). Override with **environment variables** before launching (same names as in that file: `MODEL`, `DPO_DATASETS`, `NUM_STEPS_DPO`, `TARGET_STEP_DPO`, eval knobs, etc.).

**Slurm / resources:** GPU stages use `#SBATCH --time=07:45:00` (under a typical **8h** cap). `TRAIN_TIMEOUT_PER_DATASET`, `MASK_TIMEOUT`, and `SPARSE_TIMEOUT_PER_MASK` default to **7h30m** so processes get SIGTERM before Slurm. CPU-only stage **3a** (Jaccard / CSV / plots) uses `PIPELINE_CPU_COMPARISON_TIME` (default **7h45m**), `PIPELINE_CPU_COMPARISON_MEM` (**128G**), `PIPELINE_CPU_COMPARISON_CPUS` (**16**) to cover the expanded comparison set while sparse runs elsewhere. Stage **3b** (optional CKA) uses **7h45** GPU wall in `pipeline_stage_03b_cka_gpu.sh`. **CPU vs GPU partition names:** defaults target **Northeastern Explorer** — `CPU_PARTITION` defaults to **`short`** (Explorer has no partition named `cpu`); `GPU_PARTITION` defaults to **`gpu`**. [`resume_pipeline_from_stage.sh`](resume_pipeline_from_stage.sh) passes these on the `sbatch` command line so they override embedded `#SBATCH` lines. On clusters that use a literal `cpu` CPU partition, run `export CPU_PARTITION=cpu` before submitting.

**Cancelled jobs (debug):** distinguish wall time vs policy vs OOM with accounting, e.g.  
`sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,MaxRSS,ReqGRES`  
If `Elapsed` ≈ `Timelimit`, increase wall time or rely on the **split stage 2** so each phase has its own allocation. If the job dies with GPU idle underuse, split stage 2 avoids holding a GPU during CPU-only warm/post work.

**Resume after a failure:** [`resume_pipeline_from_stage.sh`](resume_pipeline_from_stage.sh) —  
`bash scripts/resume_pipeline_from_stage.sh <stage> <PIPELINE_RUN_ID>`  
Stages: **`2`** or **`2a`** (warm on GPU, starts split chain), **`2b`** (cold Fisher+CAV), **`2c`** (random+inverses, then chains stage 3), **`2all`** (entry `pipeline_stage_02_masks.sh` → submits 02a), **`3`**–**`5`** as before.

### Single-GPU dense DPO — Tulu3 paper-style hyperparams (`SEQ` 1024)

The paper’s global batch 128 used **8 GPUs** (per-device 1 × grad-accum 16 × 8). On **one GPU**, match global batch 128 with **per-device 1 × grad-accum 128** (or any product \(=128\)).

Optional env vars are read by `run_dense_dpo()` in [`pipeline_common.sh`](pipeline_common.sh): `DPO_PER_DEVICE_TRAIN_BATCH_SIZE`, `DPO_GRADIENT_ACCUMULATION_STEPS`, `DPO_LEARNING_RATE`, `DPO_WARMUP_RATIO`, `DPO_WEIGHT_DECAY`, `DPO_MAX_LENGTH`, `DPO_MAX_PROMPT_LENGTH`, `DPO_BETA`. Gradient checkpointing defaults **on**; set `DPO_GRADIENT_CHECKPOINTING=0` to disable.

**Chained full pipeline** (stage 1 uses these DPO settings):

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"

export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export DPO_DATASETS="tulu3"
export NUM_STEPS_DPO=250

# Paper-style (Tulu3 DPO); sequence length 768 for faster steps than 2048, reduce grad accumulation from 128 to 64
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=1
export DPO_GRADIENT_ACCUMULATION_STEPS=64
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_WEIGHT_DECAY=0.0
export DPO_MAX_LENGTH=768
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
# export PIPELINE_CPU_COMPARISON_TIME=07:45:00      # stage 3a wall (default 7h45; ≤8h cluster cap)
# export PIPELINE_SKIP_SPARSE_LAUNCH=1              # resume stage 3 without re-submitting stage 4 if .sparse_launch_submitted exists
# export RANDOM_MASK_SEED=42                        # seed for random baseline mask (default 42); must match when resuming comparisons
# export PIPELINE_GENERATE_INVERSE_MASKS=0          # omit complement masks (*_inverse.pt) + their comparisons/sparse jobs (default 1)
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
- Masks: `MASK_OUT_BASE/${PIPELINE_RUN_ID}/` (includes `random_baseline_*_seed${RANDOM_MASK_SEED}.pt` and `*_inverse.pt` complement masks unless disabled)
- Comparisons: `MASK_OUT_BASE/${PIPELINE_RUN_ID}/comparisons/` — artifacts are `jaccard_<tag>.json`, optional `cka_<tag>.json`, `layer_metrics_<tag>.csv`. Tags per sparsity prefix `sp<pct>_`: **anchors** `wm_vs_cf`, `wmom_vs_cc`, `wf_vs_cf`; **random vs structured** `rand_vs_wm|wf|cf|wmom|cc`; **warm×warm** `wm_vs_wmom`, `wm_vs_wf`, `wmom_vs_wf`; **cold** `cf_vs_cc`; **cross** `wm_vs_cc`, `wf_vs_cc`, `wmom_vs_cf`; **complement** `wm_vs_wminv`, …, `wminv_vs_cf`, … (see `run_mask_comparisons` in `pipeline_common.sh`).
- Sparse launch marker: `MASK_OUT_BASE/${PIPELINE_RUN_ID}/.sparse_launch_submitted` is created after stage 4 successfully queues sparse GPU jobs (skips duplicate early launch on resume).
- Sparse runs: `SPARSE_OUT_BASE/${PIPELINE_RUN_ID}/<mask_stem>/`
- Parallel sparse **per-job logs**: `logs/sparse_${PIPELINE_RUN_ID}_<mask_stem>_*.out`
- Eval harness: `EVAL_OUT_BASE/${PIPELINE_RUN_ID}/...` (plus each eval `sbatch` log under `logs/`)

**Requirements:** nested `sbatch` (submitting the next stage from inside a running job) must be allowed. Edit `#SBATCH` lines (`partition`, `gres`, `mem`, `time`) in `pipeline_stage_*.sh`, `pipeline_sparse_one_mask.sh`, and `run_full_pipeline.sh` to match your cluster.

**Slurm submit directory:** Always run `sbatch` or `bash scripts/submit_pipeline_chain.sh` from the **repository root** (the directory you `cd` into before submitting). Slurm sets `SLURM_SUBMIT_DIR` to that path; the pipeline uses it to find `scripts/pipeline_common.sh`. If you submit from elsewhere, sourcing `pipeline_common.sh` fails with “No such file” under `/var/spool/slurmd/...`.

### Single long job (nested Slurm sparse workers)

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"   # if needed
sbatch scripts/run_full_pipeline.sh
```

This allocation runs dense DPO, masks, then **submits** parallel sparse GPU jobs (`pipeline_stage_04_sparse.sh`) and runs mask comparisons in the same job (overlap). Evals are queued by the stage 4 launcher when sparse jobs finish. Nested `sbatch` must be allowed. For the **chained** multi-job pipeline (recommended for long runs), use `submit_pipeline_chain.sh` instead.

## H200 BSR DPO benchmark (throughput and random masks)

This workflow is **separate from** the main pipeline’s `sparse_dpo_efficiency.py` stage-4 jobs. It exercises [`src/full_training/sparse_dpo_bsr.py`](../src/full_training/sparse_dpo_bsr.py) (BSR sparse backward + `SparseAdamW`) for **timing comparisons**: one **dense** phase (standard `nn.Linear`, dense AdamW, no BSR injection) and several **sparse** phases with **random global masks** at fixed target sparsities (90%, 95%, 97.5%, 99%, 99.75% by default).

- **Dataset / tokenizer** are loaded **once**; each phase **reloads the base model** from the Hub so timings are not chained across phases.
- **No W&B** (set `WANDB_MODE=disabled` in the batch script). **No delta checkpoints** and **no final model save** in the benchmark driver.
- **CSV log:** `<output_dir>/benchmark_training_log.csv` — per-step metrics plus `phase`, `cumulative_steps_per_s`, `cumulative_samples_per_s`, and Trainer fields (loss, etc.). Temporary mask `.pt` files are written under `$TMPDIR` and deleted after each sparse phase (bool masks via `save_masks`).

**Entry point:** [`src/full_training/h200_sparse_dpo_bsr_benchmark.py`](../src/full_training/h200_sparse_dpo_bsr_benchmark.py)  
**Slurm wrapper:** [`h200_sparse_dpo_bsr_benchmark.sh`](h200_sparse_dpo_bsr_benchmark.sh) (defaults: `gpu` partition, `gres=gpu:h200:1`, 128G RAM, 4h wall — adjust `#SBATCH` to match your site).

Default training knobs align with the **1024-seq Tulu3-style** block at the end of this file (`NUM_STEPS_DPO=100` per phase in the script env, `DPO_LEARNING_RATE=5e-7`, batch `2` × grad-accum `64`, warmup ratio `0.1`, max prompt/response length `1024`). Override with the same variable names the shell script exports.

### Submit (copy/paste, login node)

Run from the **repository root**. Set `HF_TOKEN` if the model is gated (Llama 3.1 8B Instruct).

```bash
cd /path/to/rl_casino
mkdir -p logs

export HF_TOKEN="hf_xxxxxxxx"   # required for meta-llama/Llama-3.1-8B-Instruct
export TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda_envs/rl_casino}"
export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"

# Optional: where to write CSV + Slurm log context (default uses SLURM_JOB_ID when scheduled)
export H200_BSR_OUT="${SCRATCH_USER_ROOT}/rl_casino_h200_bsr/run_${SLURM_JOB_ID:-manual}"

sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/h200_bsr_bench_<JOBID>.out
```

**Outputs:** CSV at `$H200_BSR_OUT/benchmark_training_log.csv` (or the path printed in the job log). Per-phase HF run folders also appear under `$H200_BSR_OUT/` with names like `h200_bsr_<jobid>_phase_dense/` (checkpoints disabled; mostly empty dirs).

### Plots from the CSV

After the job finishes (or download the CSV to your laptop):

```bash
cd /path/to/rl_casino
python scripts/plot_h200_bsr_benchmark.py \
  --csv /scratch/${USER}/rl_casino_h200_bsr/run_<JOBID>/benchmark_training_log.csv \
  --out_dir /scratch/${USER}/rl_casino_h200_bsr/run_<JOBID>/plots
```

Writes PNGs: `h200_bsr_benchmark_throughput_samples.png`, `h200_bsr_benchmark_throughput_steps.png`, `h200_bsr_benchmark_loss.png`, `h200_bsr_benchmark_summary_bars.png`. Requires **pandas** and **matplotlib** (see repo `requirements.txt`).

### Manual run (no Slurm)

```bash
cd /path/to/rl_casino
export HF_TOKEN="hf_xxxxxxxx"
export PYTHONPATH=/path/to/rl_casino
"$TRAIN_ENV/bin/python" src/full_training/h200_sparse_dpo_bsr_benchmark.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir /scratch/${USER}/h200_bsr_manual \
  --dataset_cache_dir /scratch/${USER}/hf_cache/datasets \
  --device_map none
```

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
J2=$(sbatch --parsable --dependency=afterok:"$J1" --export=ALL,PIPELINE_RUN_ID="$RUN_ID",RUN_ID="$RUN_ID" scripts/pipeline_stage_02a_masks_warm.sh)
echo "Stage 2 (split masks: 02a→02b→02c) first job id: $J2"

# Stage 2 chains stage 3 (comparisons): stage 3 submits stage 4 (sparse) at job start, then runs comparisons; stage 5 evals queue when sparse jobs finish.
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

### Reference: DPO hyperparameters (1024 context, global batch 128 on one GPU)

```bash
export NUM_STEPS_DPO=500
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_MAX_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=1024
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
export DPO_GRADIENT_ACCUMULATION_STEPS=64
export DPO_GRADIENT_CHECKPOINTING=1
```

The [H200 BSR benchmark](#h200-bsr-dpo-benchmark-throughput-and-random-masks) section uses the same style of exports via `h200_sparse_dpo_bsr_benchmark.sh` (default `NUM_STEPS_DPO=100` per phase for quick throughput measurement).
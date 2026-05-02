# Open-R1 Math GRPO — Llama 3.1 8B (runbook)

This document describes the **intended production path** for large-scale GRPO on **Open-R1 Math** with **`meta-llama/Llama-3.1-8B-Instruct`**: Slurm launcher [`scripts/grpo_openr1_llama31_slurm.sh`](../scripts/grpo_openr1_llama31_slurm.sh), dense training via [`src/full_training/GRPO_train.py`](../src/full_training/GRPO_train.py), optional sparse BSR via [`src/full_training/sparse_grpo_bsr.py`](../src/full_training/sparse_grpo_bsr.py).

## Canonical hyperparameters

All defaults, DPO benchmark reference numbers from `scratch.md`, and rationale live in **[`docs/hyperparams/open_r1_llama31.yaml`](hyperparams/open_r1_llama31.yaml)**. Update that file when you change defaults in the Slurm script or training CLIs.

## Quick launch (dense, single GPU)

From repo root (so `SLURM_SUBMIT_DIR` and `logs/` resolve correctly):

```bash
export HF_TOKEN="hf_..."   # if Llama is gated
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export GRPO_MODE=dense
export GRPO_NGPUS=1
export GRPO_TARGET_STEPS=1000
# Optional: fixed folder for resume — set on first job and reuse on requeue
# export GRPO_RUN_SLUG="llama31_math220k_grpo_dense"
# export GRPO_RUN_NAME="my_run"
sbatch scripts/grpo_openr1_llama31_slurm.sh
```

The script includes **`#SBATCH --gres=gpu:h200:1`** (same pattern as `h200_sparse_dpo_bsr_benchmark.sh`). **Do not remove `--gres`** when using the `gpu` partition on Northeastern Explorer—submitting without a GPU request can produce **`Access/permission denied`**. If your site uses a different GRES string, edit that one line (e.g. `--gres=gpu:1`).

## Sparse BSR

Requires a compatible mask `.pt`:

```bash
export GRPO_MODE=sparse
export GRPO_MASK=/path/to/mask.pt
sbatch scripts/grpo_openr1_llama31_slurm.sh
```

Sparse mode uses **`--run_name`** (not `--run_slug`) for stable output folders; set **`GRPO_RUN_NAME`** before the first job if you need deterministic resume paths.

## Resume

- **Dense:** pass **`GRPO_RESUME=auto`** and the **same `GRPO_RUN_SLUG`** (and model/dataset) as the first job. Checkpoint and `wandb_run_id.txt` live under the run directory.
- **Sparse:** **`GRPO_RESUME=auto`** with the same **`GRPO_RUN_NAME`** and output base.

### `GRPO_SAVE_STEPS` and `GRPO_SAVE_TOTAL_LIMIT`

These map to Hugging Face `TrainingArguments` / `GRPOConfig`: **`save_strategy="steps"`** plus **`save_steps`** and **`save_total_limit`**.

| Variable | Meaning |
|----------|--------|
| **`GRPO_SAVE_STEPS`** | Every **`N` global steps**, the trainer writes a checkpoint under the run’s checkpoint directory (e.g. `run_dir/checkpoints/checkpoint-50`, `checkpoint-100`, …). Smaller **N** = more frequent disk writes and more recovery points if the job dies; slightly more I/O overhead. |
| **`GRPO_SAVE_TOTAL_LIMIT`** | Keep at most **`K` saved checkpoints** on disk. Older checkpoints are **deleted** as new ones are written (a rotating window). This caps scratch usage; it does **not** cap how many steps you train — only how many historical snapshots you retain. Example: with `K=3`, you might keep `checkpoint-900`, `checkpoint-950`, `checkpoint-1000` after a 1000-step run (exact names depend on step alignment). |

**Resume** uses the **latest** checkpoint when you pass `--resume_from_checkpoint auto` (see `resolve_resume_checkpoint` in `src/utils/grpo_checkpoint_utils.py`). You only need **one** valid checkpoint on disk to continue; the limit mainly affects how far **back** you can roll manually.

### Automatic Slurm requeue

**No.** `scripts/grpo_openr1_llama31_slurm.sh` does **not** submit a follow-up job, chain with `sbatch --dependency=afterok`, or enable Slurm’s `--requeue`. If the job hits **wall time**, **preemption**, or **OOM**, training stops when the process is killed.

**What to do:** Ensure **`GRPO_SAVE_STEPS`** is small enough that you get a checkpoint **before** the typical stop point (e.g. every 50 steps with an 8h wall). After the job ends, submit again with the **same** `GRPO_RUN_SLUG` / `GRPO_RUN_NAME` and:

```bash
export GRPO_RESUME=auto
export GRPO_TARGET_STEPS=5000   # same final goal as before
sbatch scripts/grpo_openr1_llama31_slurm.sh
```

The trainer will load the latest `checkpoint-*` and continue until `max_steps` is reached. For chained automation, wrap that in your own script or use Slurm dependencies (similar to `scripts/submit_pipeline_chain.sh` for the DPO pipeline).

## GRPO reward profile (`GRPO_REWARD_PROFILE`)

Math rewards are defined in [`src/utils/grpo_rewards.py`](../src/utils/grpo_rewards.py).

| Profile | When to use |
|---------|-------------|
| **`llama_cot`** (default) | Instruct models (e.g. Llama 3.1) that rarely emit `<redacted_thinking>` tags. Uses delimiter heuristics (`####`, `\boxed{}`, `Final Answer:`) so `format_reasoning` and `format_number` are not degenerate. |
| **`openr1_tags`** | Strict OpenR1-style blocks only; prompts get an optional suffix asking for redacted tags (dense/sparse GRPO loaders). |

Override: `export GRPO_REWARD_PROFILE=openr1_tags` before `sbatch`, or pass `--grpo_reward_profile` if invoking Python directly.

### Evaluation parity (benchmarks)

- Use the **same** [`run_evals_slurm.sh`](../scripts/run_evals_slurm.sh) / lm-eval settings and **`apply_chat_template`** behavior as your **base** model for fair accuracy comparisons.
- Training caps completions at **`GRPO_MAX_COMPLETION_LENGTH`** (default 2048). If lm-eval tasks use a **much lower** `max_gen_toks`, long-CoT checkpoints may look weaker than they are — align generation limits when comparing to GRPO-tuned checkpoints (or report two caps).
- Standard math harnesses extract final answers from full model output; they do **not** require training-time reward tags.

## Environment: CUDA

Training **must** run where **`torch.cuda.is_available()`** is true. On **login nodes** or CPU partitions, PyTorch often loads **fp32** weights and the process can be **OOM-killed** with no Python traceback.

The Slurm script runs a **CUDA preflight** before training. For interactive checks:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

## Key environment variables (override defaults)

| Variable | Role |
|----------|------|
| `MODEL` | HF model id |
| `GRPO_DATASET` | Registry key (default `math-220k`) |
| `GRPO_LR`, `GRPO_BETA` | LR and GRPO β |
| `GRPO_PER_DEVICE_BS`, `GRPO_GRAD_ACCUM` | Batch |
| `GRPO_NUM_GEN`, `GRPO_GEN_BATCH` | Rollouts (gen_batch divisible by num_gen) |
| `GRPO_MAX_PROMPT_LENGTH`, `GRPO_MAX_COMPLETION_LENGTH` | Sequence caps |
| `GRPO_PRECISION` | `auto` \| `bf16` \| `fp16` (default **`bf16`** in launcher for H200-class GPUs) |
| `GRPO_USE_WANDB` | `1` (default) adds `--use_wandb`; set `0` for offline/disabled UI |
| `GRPO_RESUME` | empty, `auto`, or path to `checkpoint-*` |
| `GRPO_SAVE_STEPS`, `GRPO_SAVE_TOTAL_LIMIT` | Checkpoint frequency and how many checkpoints to retain on disk (see section above) |
| `GRPO_REWARD_PROFILE` | `llama_cot` (default) or `openr1_tags` — see [GRPO reward profile](#grpo-reward-profile-grpo_reward_profile) |
| `TRL_SKIP_VLLM_IMPORT` | default `1` — see [`TROUBLESHOOTING_GRPO.md`](TROUBLESHOOTING_GRPO.md) |

## Related docs

- [`TROUBLESHOOTING_GRPO.md`](TROUBLESHOOTING_GRPO.md) — vLLM import, precision, bf16 on older GPUs
- [`scripts/README.md`](../scripts/README.md) — Slurm submission conventions

# Qwen3-8B high-sparsity DPO treatment — full faithful chain

**Date prepared:** 2026-06-01
**Branch:** `irene-sparse-speed-ablation`
**Author:** Irene (via Claude)
**Goal:** Test whether the high-sparsity (97.5%) oracle-mask DPO finding from
Llama-3.1-8B generalizes to an *alternative* backbone — **Qwen3-8B**.

## Model (read this)
- Backbone: **`Qwen/Qwen3-8B`** — the post-trained / instruct chat model.
- **NOT** `Qwen/Qwen3-8B-Base` (pretrained base). There is no separate text-only
  `Qwen/Qwen3-8B-Instruct` repo; the plain `Qwen/Qwen3-8B` *is* the instruct model.
- Qwen3 uses Llama-style param names (q/k/v/o_proj, gate/up/down_proj), so
  `DPO_train.py`, `checkpoint_diff_mask_finder.py`, and `sparse_dpo_efficiency.py`
  (all AutoModel/`named_parameters`-based) work unchanged.

## Design — mirrors the Llama transfer pipeline exactly (Scott's algo + hparams)
Two parallel dependency chains (light-r1 and tulu3):

```
p1 dense DPO  --afterok-->  p2 oracle ckpt-diff mask @97.5%  --afterok-->  p3 sparse DPO
```

| Phase | Script | Notes |
|---|---|---|
| 1 | `scripts/qwen3_p1_dense_dpo.sbatch` | dense DPO, saves ckpt-150 + ckpt-500 |
| 2 | `scripts/qwen3_p2_oracle_dpo.sbatch` | oracle = top-2.5% of \|θ_500 − θ_0\|, min_layer_keep 0.0025 |
| 3 | `scripts/qwen3_p3_sparse_dpo.sbatch` | sparse DPO w/ step-500 oracle mask + SparseAdamW |
| — | `scripts/qwen3_transfer_launch.sh` | submits all 6 jobs with afterok deps |

## Hyperparameters (frozen, identical to Llama transfer p1/p3)
| Param | Value |
|---|---|
| Datasets | light-r1, tulu3 |
| Steps | 500 (dense + sparse) |
| Eff. batch | 128 (per_device 2 × grad_accum 64 × 1 H200) |
| lr | 5e-7 |
| dpo_beta | 0.1 |
| warmup_ratio | 0.1 |
| max_length / max_prompt_length | 1024 / 1024 |
| weight_decay | 0.0 |
| Sparsity (ρ, zeroed fraction) | **97.5%** |
| min_layer_keep_ratio | 0.0025 |
| Optimizer (sparse phase) | `sparse_adamw` (mask-aware; does not decay frozen weights) |
| gradient_checkpointing | on |
| Hardware | 1× H200, `--time=08:00:00 --requeue` (school 8h cap) |

## Output paths (/scratch/$USER)
- Dense ckpts: `/scratch/xie.yiyi/transfer_v1/dense_dpo_{light_r1,tulu3}_qwen3_8b/`
- Oracle masks: `/scratch/xie.yiyi/transfer_v1/oracle_masks_qwen3_8b/oracle_dpo_{light_r1,tulu3}_step{500,150}_sp97.5.pt`
- Sparse runs: `/scratch/xie.yiyi/transfer_v1/sparse_dpo_{light_r1,tulu3}_qwen3_8b_oracle_step500/`
- WandB project: `rl_casino_transfer_v1`

## LAUNCHED 2026-06-01

Submitted via `scripts/qwen3_transfer_launch.sh`. Job IDs:

| Job | Phase | Dataset | Dependency |
|---|---|---|---|
| 7362092 | p1 dense DPO | light-r1 | — |
| 7362094 | p2 oracle mask | light-r1 | afterok:7362092 |
| 7362097 | p3 sparse DPO | light-r1 | afterok:7362094 |
| 7362100 | p1 dense DPO | tulu3 | — |
| 7362102 | p2 oracle mask | tulu3 | afterok:7362100 |
| 7362105 | p3 sparse DPO | tulu3 | afterok:7362102 |

Connection note: the `Discovery_Cluster` alias (hardcoded IP `129.10.0.146`) was
refusing connections — that login node was down. Used
`login.explorer.northeastern.edu` (round-robins to `.145`/`.146`) instead;
key-based auth works fine. Scripts were `scp`'d to the worktree (not committed).
`Qwen/Qwen3-8B` verified non-gated, downloads on first run.

Watch: `ssh login.explorer.northeastern.edu 'squeue -u $USER'`

## Caveats / things to watch
- **Qwen3 thinking-mode chat template.** `Qwen/Qwen3-8B` defaults to a template
  that can inject `<think>` blocks. DPO formatting goes through the tokenizer's
  chat template; verify the formatted pairs look sane on the first dense run
  (check the early WandB samples / stdout) before trusting downstream numbers.
- DPO loads policy + frozen ref model → ~2× weights in memory; Llama-8B fit on
  1 H200 at these settings, Qwen3-8B (≈8.2B) is comparable, but watch for OOM on
  the first dense step (supervisor-style fix: halve per_device, double grad_accum).
- Sparsity convention: ρ = fraction zeroed = 97.5% → only 2.5% of weights train.

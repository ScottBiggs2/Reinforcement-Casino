# GRPO warm-start T-sweep — dense trajectory, mask family, comparison, sparse arms

**Date:** 2026-07-26 · **Branch:** `irene-rebuttal-lora` · **Worktree:** `/home/xie.yiyi/rc-sparse-speed`
**Purpose:** AC meta-review item 3 (ablations over scoring functions) and reviewer bf6h W1.
The paper answers "how early is the subnetwork decided" for DPO (warm-magnitude
k = 50/100/150/200 vs oracle, Fig. 5) and never answers it for GRPO.

---

## 0. What already existed, and why it was not enough

| Asset | Status |
|---|---|
| `transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt` | exists, 2026-04-29 |
| `transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step850_sp97.5.pt` | exists, 2026-04-29 |
| `probe_variance/masks/cold_cav_v2_grpo_seed42.pt` | exists (CAV, different family) |
| dense GRPO trajectory `transfer_v1/dense_grpo_math220k_llama8b/llama8b_math220k_dense_scott_full` | ran to **step 850**, `max_steps` scheduled 5000 |
| checkpoints of that run | **only 500, 550, …, 850** |

So GRPO oracle masks exist, but **no warm-start magnitude mask at any T is buildable**,
because no dense GRPO checkpoint below step 500 survives. Two independent causes, both
verified rather than assumed:

1. `GRPO_train.py` defaults `--save_total_limit 3` — a rolling three-checkpoint window.
2. That run's own `run_manifest.json` records
   `resume_from_checkpoint=.../checkpoint-300`, i.e. steps 50–300 existed and are gone.

The only sub-500 GRPO checkpoints anywhere on scratch belong to
`dense_grpo_math220k_smoke_lc3072` (steps 25/50/75/100), which is a **3072-cap smoke run** —
a different trajectory from the one the oracle was built on, so unusable here.

## 1. Why a fresh trajectory rather than replaying the old one

The old dense run was `lr_scheduler_type=cosine, max_steps=5000, warmup_ratio=0.1`, i.e.
a **500-step warmup**. T = 50/100/250 would all land inside the LR ramp at 10%/20%/50%
of peak, so "the mask has stabilised by T" would be confounded with "the LR has not
finished ramping". That is the same defect documented in
[grpo_schedule_confound_2026-07-25.md](grpo_schedule_confound_2026-07-25.md).

This sweep instead uses the schedule-clean configuration from
`scripts/grpo_matched_schedule.sbatch` (cosine over `max_steps=500`, `warmup_ratio=0.1`
→ 50-step warmup) and runs the full 500 steps, so **one self-consistent trajectory
yields both the oracle (T=500) and every warm-start point**.

## 2. Job chain

| Job | Script | Depends on | Partition | Resources |
|---|---|---|---|---|
| **8760188** | `scripts/grpo_tsweep_dense.sbatch` | — | multigpu | `gpu:h200:1`, 192G, 12 cpu, 8h, `--requeue` |
| **8760223** | `scripts/grpo_tsweep_masks.sbatch` | `afterok:8760188` | short | CPU only, 192G, 16 cpu, 6h |
| **8760224** | `scripts/grpo_tsweep_compare.sbatch` | `afterok:8760223` | gpu | `gpu:1`, 192G, 8 cpu, 3h |
| **8760259** | `scripts/grpo_tsweep_launch_sparse.sbatch` | `afterok:8760223` | short | 1 cpu, 2G, 12h |
| ↳ submits | `scripts/grpo_tsweep_sparse.sbatch` | (launcher) | multigpu | `gpu:h200:1`, 192G, 12 cpu, 8h, array 0–3 |

The launcher exists because the multigpu QOS caps a user at **8 submitted jobs** and the
LoRA baseline array (`8727274`, elements 0–3, rebuttal item 2) already holds four. Direct
submission of the 4-arm sparse array returned `QOSMaxSubmitJobPerUserLimit`. The launcher
sits on the uncapped `short` QOS and retries every 10 min until a slot frees. **Nothing
already queued was cancelled.**

## 3. Hyperparameters — dense (job 8760188)

| field | value | source |
|---|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct` | locked model |
| dataset | `open-r1/OpenR1-Math-220k` (`math-220k`) | paper |
| num_steps | **500** | paper parity |
| learning_rate | 5e-6 | paper |
| lr_scheduler_type | **cosine** | matched-schedule config |
| warmup_ratio | **0.1** (→ 50 steps) | matched-schedule config |
| beta (KL) | 0.025 | paper |
| max_grad_norm | 0.1 | paper |
| num_generations | 8 | paper |
| generation_batch_size | 8 | paper |
| per_device_train_batch_size | 2 | paper |
| gradient_accumulation_steps | 4 | paper |
| max_prompt_length | 512 | paper |
| max_completion_length | **1024** | **as-run value, deliberately unchanged** |
| grpo_reward_profile | `llama_cot` | matches old run manifest |
| optim | `adamw_8bit` (default) | paper |
| precision | auto → bf16 | paper |
| **save_steps** | **50** | *changed* — the whole point |
| **save_total_limit** | **20** | *changed* — default 3 is what destroyed the old run |
| resume_from_checkpoint | `auto` | 8h walltime cap + `--requeue` |

Output: `/scratch/xie.yiyi/rebuttal_analysis/grpo_tsweep/dense/grpo_tsweep_dense_500steps/`
· wandb project `rl_casino_rebuttal`, run `grpo_tsweep_dense_500steps_cap1024`.

**Cost basis:** job `8734159_0` measured **44:09 for 150 steps** on 1×H200 at cap 1024
→ ~17.6 s/step → ~2.5 h for 500 steps plus ten 34 GB checkpoint writes (~340 GB scratch).

## 4. Mask generation (job 8760223)

`src/warm_start/checkpoint_diff_mask_finder.py`, ρ = **97.5%**, global pooling (no
`--local_pool`), `DEFAULT_MIN_LAYER_KEEP_RATIO`, element-wise, `--force_cpu`.

| T | kind | output |
|---|---|---|
| 50 | warm-start magnitude | `grpo_math220k_warmmag_T50_sp97.5.pt` |
| 100 | warm-start magnitude | `grpo_math220k_warmmag_T100_sp97.5.pt` |
| 250 | warm-start magnitude | `grpo_math220k_warmmag_T250_sp97.5.pt` |
| 500 | **oracle** (k = T) | `grpo_math220k_oracle_T500_sp97.5.pt` |

Into `/scratch/xie.yiyi/rebuttal_analysis/grpo_tsweep/masks/`.

**Mask-finder version is a deliberate choice.** This branch's version is used, *not*
`origin/cav_fixes`. Scott's branch has since (a) restricted scoring to 2-D `weight`
tensors, excluding 1-D norms, (b) added tied-parameter detection — a no-op for
Llama-3.1-8B, whose `tie_word_embeddings` is `false` — and (c) dropped the seed-jitter
path. Scott's 2-D restriction is arguably the better definition and matches the
random-mask universe (it is the same issue as the "2.5% of 2-D matrices, not of all
parameters" disclosure in AC item 4d). But it is **not** the code that produced
`oracle_grpo_math220k_step500_sp97.5.pt`. Keeping this branch's version makes the new
T=500 oracle directly comparable to the paper's, which is worth more here.

## 5. Comparison (job 8760224)

`scripts/multi_mask_jaccard_cka.py` over six masks → 15 pairs: global + per-layer
Jaccard, linear CKA (`down_proj`, 64 calibration samples from tulu3, seed 42), plus
`src/analysis/mask_composition.py` for the tensor-class audit.

Two controls are in the set on purpose:

- **Random-seed42** — the chance floor. App. A.6.1 gives E[J] = A/(2−A) ≈ 1.27e-2 at
  ρ = 97.5%, so every T-vs-oracle Jaccard is read against it.
- **GRPO-oracle-paper500** — the oracle already in the paper, same mask-finder code,
  same objective/dataset/sparsity, **different LR schedule**. New-oracle vs paper-oracle
  is a free read on how much of the "oracle subnetwork" is schedule artifact.

Output: `/scratch/xie.yiyi/rebuttal_analysis/grpo_tsweep/compare/`.

## 6. Sparse training arms (array submitted by 8760259)

All four schedule-matched to the dense run: cosine, 500 steps, 50-step warmup, lr 5e-6,
β 0.025, cap 1024, `--optimizer sparse_adamw`, `llama_cot` rewards. No checkpoints
written (`--save_model false`); the deliverable is the wandb reward curve.

| arm | mask |
|---|---|
| 0 | oracle T=500 |
| 1 | warm-magnitude T=250 |
| 2 | warm-magnitude T=100 |
| 3 | random seed42 |

Because these share one schedule with job 8760188, a gap between them is attributable to
the mask — which is exactly what Appendix D.4 could not claim (dense sat at 99.8% of peak
LR at step 500 while every sparse arm had annealed to 0.2%, a 449× gap).

## 7. Not touched

- `8727274` LoRA baseline array (item 2) — left running.
- `8734159_1` sparse arm of the 150-step matched-schedule test with the **old** oracle
  mask — left queued; it answers the narrower D.4-confound question and is cheap.
- Existing masks under `transfer_v1/oracle_masks_llama8b/` — read-only here.

## 8. Results

_Pending._ Fill in when 8760188 → 8760223 → 8760224 land.

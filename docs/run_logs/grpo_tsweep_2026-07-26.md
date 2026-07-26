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
| **8761309** | `scripts/grpo_tsweep_dense.sbatch` | — | gpu,multigpu | `gpu:h200:1`, 192G, 12 cpu, 8h, `--requeue` |
| **8761310** | `scripts/grpo_tsweep_masks.sbatch` | `afterok:8761309` | short | CPU only, 192G, 16 cpu, 6h |
| **8761311** | `scripts/grpo_tsweep_compare.sbatch` | `afterok:8761310` | gpu | `gpu:1`, 192G, 8 cpu, 5h |
| **8761312** | `scripts/grpo_tsweep_launch_sparse.sbatch` | `afterok:8761310` | short | 1 cpu, 2G, 12h |
| ↳ submits | `scripts/grpo_tsweep_sparse.sbatch` | (launcher) | gpu,multigpu | `gpu:h200:1`, 192G, 12 cpu, 8h, array 0–3 |

**Partition:** the only h200 nodes (d4052–d4055) belong to *both* `gpu` and `multigpu`,
and the two partitions carry separate QOS counters (each 8 submitted / 4 running). The
GRPO jobs list both, which doubles the scheduling chances on identical hardware; `gpu`
caps at 8h, already the walltime here, so nothing is given up. Applied to 8761309 and
8734159_1 in place via `scontrol update Partition=...` — no resubmission.

**Queue made room for this (2026-07-26, Irene's call):** LoRA baseline arms
`8727274_1` (r=64, lr 3e-5) and `8727274_3` (r=16, lr 3e-5) were cancelled while still
PENDING — never started, no work lost. Reason: arm 0 (r=64, lr 5e-6, the *lowest* grid
point) had already reached train margin **10.44** with accuracy **1.00** and loss 2e-4 at
step 400/500, against dense DPO's 3.24. On Light-R1 (~3,060 pairs) 500 steps at effective
batch 128 is ≈21 epochs, so every LR in the grid will read accuracy 1.00 and the grid
cannot be adjudicated on training-batch metrics at all — it only becomes meaningful once
the held-out preference eval runs. `8727274_0` was left running (nearly done) and
`8727274_2` (r=64, lr 1e-4) kept, because that arm tests likelihood displacement
(Razin et al., ICLR 2025) via `logps/chosen` vs `logps/rejected`, which is a separate
question from LR selection and does not depend on the margin.

An earlier chain (8760188/8760223/8760224/8760259) was submitted at `cap=1024`, then
cancelled and resubmitted at `cap=2048` before any of it started — see §3a. Nothing else
was cancelled.

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
| max_completion_length | **2048** | **Scott's locked spec + Table 6; see §3a** |
| grpo_reward_profile | `llama_cot` | matches old run manifest |
| optim | `adamw_8bit` (default) | paper |
| precision | auto → bf16 | paper |
| **save_steps** | **50** | *changed* — the whole point |
| **save_total_limit** | **20** | *changed* — default 3 is what destroyed the old run |
| resume_from_checkpoint | `auto` | 8h walltime cap + `--requeue` |

Output: `/scratch/xie.yiyi/rebuttal_analysis/grpo_tsweep/dense/grpo_tsweep_dense_500steps/`
· wandb project `rl_casino_rebuttal`, run `grpo_tsweep_dense_500steps_cap1024`.

**Cost basis:** job `8734159_0` measured **44:09 for 150 steps** on 1×H200 at cap 1024
→ ~17.6 s/step. At cap 2048 budget ~4.5–5 h for 500 steps plus ten 34 GB checkpoint
writes (~340 GB scratch).

## 3a. Alignment with Scott's locked GRPO spec

Checked against `docs/hyperparams/grpo_500step_5way_sweep.yaml` on `origin/cav_fixes`
(`updated: 2026-05-01`), which is the newest GRPO specification anywhere in the project.
**Scott's last GRPO-touching commit is 2026-05-04** (`ee21ab9`); everything on `cav_fixes`
from 2026-05-05 to its tip (`c742781`, 2026-06-21) is Qwen3-8B/32B DPO, sparse-kernel
speed benchmarking, and the 32B NaN fix. There is no GRPO training newer than the paper's.

Every locked field matches this run — model, dataset, 500 steps, lr 5e-6, β 0.025,
`max_grad_norm` 0.1, cosine, `warmup_ratio` 0.1 (sparse: 50 warmup steps), `adamw_8bit`
dense / `sparse_adamw` sparse, bf16, batch 2×4, 8 generations, `generation_batch_size` 8,
`max_prompt_length` 512, `llama_cot`, ρ 97.5%, element granularity, and
`min_layer_keep_ratio` 0.0025 (confirmed: `DEFAULT_MIN_LAYER_KEEP_RATIO = 0.0025` in
`src/utils/mask_utils.py:16`, which is what this run inherits by not passing the flag).

Three deliberate departures:

1. **`max_completion_length` 2048**, per Scott's spec and Table 6 — *not* the as-run 1024
   that Fig. 9's arms actually used. Chosen (Irene, 2026-07-26) because at 1024
   `completions/clipped_ratio` is 0.5–1.0 and roughly 90% of the optimised reward is
   formatting, so a T-sweep at 1024 would measure mask quality against a target that
   barely scores reasoning. Cost: dense goes from ~2.5 h to ~5 h, and the result is no
   longer point-for-point comparable to the paper's existing GRPO arms.
2. **`save_total_limit` 20** instead of Scott's 3 — the entire reason this run exists.
3. **Masks from saved checkpoints, not from delta logs.** Scott's canonical warm-magnitude
   path is `--delta_log_interval 50 --delta_log_end_step 200` feeding
   `src/warm_start/even_better_mask_finder.py --method magnitude --target_step 200`.
   Verified that this is the *same quantity*: `DeltaLoggingCallback` is constructed with
   `base_state` = θ⁰ captured before training and logs deltas **vs init**, i.e. |θ^T − θ⁰|,
   not a path length Σ|θ^t − θ^{t−1}|. So checkpoint-diff and delta-log produce the same
   mask, except that the delta logger applies `threshold=1e-5` and the checkpoint path
   does not. Checkpoint-diff is also what built every existing oracle, so it keeps the
   comparison to `oracle_grpo_math220k_step500_sp97.5.pt` honest.

Scott's spec caps its delta schedule at `end_step: 200`, so his path could not have
produced T=250 at all. Saving every 50 steps to 500 is a strict superset of both his grid
and the one requested.

## 4. Mask generation (job 8760223)

`src/warm_start/checkpoint_diff_mask_finder.py`, ρ = **97.5%**, global pooling (no
`--local_pool`), `DEFAULT_MIN_LAYER_KEEP_RATIO`, element-wise, `--force_cpu`.

| T | kind | output |
|---|---|---|
| 50 | warm-start magnitude | `grpo_math220k_warmmag_T50_sp97.5.pt` |
| 100 | warm-start magnitude | `grpo_math220k_warmmag_T100_sp97.5.pt` |
| 150 | warm-start magnitude | `grpo_math220k_warmmag_T150_sp97.5.pt` |
| 200 | warm-start magnitude | `grpo_math220k_warmmag_T200_sp97.5.pt` |
| 250 | warm-start magnitude | `grpo_math220k_warmmag_T250_sp97.5.pt` |
| 500 | **oracle** (k = T) | `grpo_math220k_oracle_T500_sp97.5.pt` |

T=150 and T=200 are included so the GRPO curve is sampled on **exactly** the grid the
submission uses for DPO (k = 50/100/150/200, Fig. 1 and Fig. 5), and so T=200 matches
`magnitude.target_step: 200` in Scott's locked yaml. Without them the two objectives
would be compared on differently-sampled curves.

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

`scripts/multi_mask_jaccard_cka.py` over eight masks → 28 pairs: global + per-layer
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

## 7a. Loose end — chased and closed

Scott's 5-way sweep (`orchestrate_grpo_500step_5way.slurm`, 2026-05-01) locks **GraSP-ABS**
and **GraSP-ABS+per-weight-SNR** GRPO arms at ρ=97.5% on this exact configuration, which
raised the question of whether item 3's scoring-function concession understates existing
coverage. **Checked 2026-07-26: it does not.** No GraSP mask was ever produced — the
orchestrator directories named `grasp` hold only random masks, the `grasp_elem_base`
directory is empty, and no file with `grasp` in its name exists in Scott's scratch. Full
evidence and root cause in [DO_NOT_REPEAT.md](DO_NOT_REPEAT.md). The item-3 concession
stands as written.

## 8. Results

_Pending._ Fill in when 8760188 → 8760223 → 8760224 land.

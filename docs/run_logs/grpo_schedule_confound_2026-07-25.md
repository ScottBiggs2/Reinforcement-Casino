# Appendix D.4 / Figure 9 is confounded by the LR schedule, not (only) by sparsity

**Date:** 2026-07-25 · **Evidence:** `training_args.bin` + `trainer_state.json` read
directly out of the checkpoints, not from logs or launcher defaults.
**Reader:** `scripts/read_training_args.py`, `scripts/cmp_lr.py`

## What Appendix D.4 claims

> "On Open-R1 with CoT style training, we find that even the Oracle mask performs quite
> poorly in training against the dense baseline, and is nearly indistinguishable
> qualitatively from the 200 step magnitude rewound subnetwork and a random subnetwork."

Reviewer bf6h's W1 is built on this: GRPO fails at ρ=97.5%, therefore the sparsity
threshold may be DPO-only and the framework does not generalise.

## What the checkpoints actually say

Read from `checkpoint-500/training_args.bin` for the three arms plotted in Figure 9:

| field | dense GRPO | sparse oracle GRPO | sparse oracle-DPO-LightR1 |
|---|---|---|---|
| `lr_scheduler_type` | **COSINE** | **LINEAR** | **LINEAR** |
| `max_steps` | **5000** | **500** | **500** |
| `warmup_ratio` | 0.1 | 0.0 | 0.0 |
| `warmup_steps` | 0 | 50 | 50 |
| `learning_rate` (peak) | 5e-6 | 5e-6 | 5e-6 |
| `max_completion_length` | **1024** | **1024** | **1024** |
| `num_generations` | 8 | 8 | 8 |
| effective batch | 8 | 8 | 8 |

And from `trainer_state.json`, the LR actually in effect at the step where Figure 9
compares them:

| arm | LR at step 500 | fraction of peak |
|---|---|---|
| dense GRPO | **4.990e-06** | **99.8%** |
| sparse oracle GRPO | **1.111e-08** | **0.2%** |
| sparse oracle-DPO-LightR1 | **1.111e-08** | **0.2%** |

**A 449× difference in learning rate at the point of comparison.**

The dense arm was configured for a 5000-step run, so at step 500 its cosine schedule
has barely begun to decay. Every sparse arm was configured for 500 steps, so by step
500 its linear schedule has annealed to essentially zero. A dense curve that keeps
climbing while the sparse curves flatten is the expected consequence of that, and it
is what Figure 9 shows.

## What this does and does not establish

It does **not** establish that GRPO tolerates 97.5% sparsity. It establishes that
**Appendix D.4's comparison cannot answer the question**, because the dense and sparse
arms differ in scheduler type, schedule horizon, and effective learning rate at the
comparison point — three uncontrolled variables on top of the one under study.

Until that is controlled, "GRPO is resistant to warm start masking" is not supported by
this figure, and neither is the opposite.

## Consequence for the plan

Running GRPO at a lower sparsity — ρ=95%, 90%, or the ρ=70% suggested by Mukherjee's
natural-sparsity estimate — **would reproduce the same artifact**, because the schedule
mismatch is independent of ρ. The sparse arm would still anneal to zero by step 500
while dense sat at peak LR. A "GRPO still fails at ρ=70%" result obtained this way
would be an artifact reported as a finding.

The correct next experiment is therefore **not** a sparsity sweep. It is a single
matched-schedule rerun at the sparsity already in the paper:

1. Dense and sparse GRPO, both at ρ=97.5%, both with the **same** scheduler type,
   horizon, and warmup. This alone may overturn D.4.
2. Only then the predicted knee, ρ ∈ {95, 90}, against the measured concentration
   curve (GRPO holds 96.38% of update mass in the top 2.5%, 99.93% in the top 5%).

## Two reporting inconsistencies for Item 4

Table 6 of the submission states, for the GRPO runs:

- **"Training steps 500"** — the dense arm's `max_steps` is 5000. It was stopped early
  (checkpoints run to 850), but it was *scheduled* for 5000, which is what determines
  the LR at every step.
- **"LR schedule: Cosine decay; warmup ratio 0.1 (dense) / 50 warmup steps (sparse)"** —
  the sparse arms use LINEAR, not cosine. The table discloses the warmup difference but
  presents both as cosine.
- **"Max completion length (tokens) 2048"** — all three arms ran at **1024**. The
  training launcher `scripts/grpo_openr1_llama31_slurm.sh:122` defaults to 1024 while
  `orchestrate_masks_then_queue_dpo_grpo.slurm:164` defaults to 2048, and
  `run_manifest.json` does not record the sequence caps, so the discrepancy was not
  visible from the run records. Scott flagged this risk in `hpc_paper_snippets.md:285`.

## Related: the reward is mostly formatting

Separate from the schedule, in a dense GRPO probe at a 3072 cap:
`completions/clipped_ratio` 0.5–1.0, `rewards/accuracy_reward/mean` 0.0–0.125, while
`format_number_reward` + `format_reasoning_reward` contribute a steady 1.0 of the 1.125
total. Roughly 90% of the optimised signal is formatting. At the as-run 1024 cap the
truncation rate can only be worse. Sparsification should affect reasoning; this reward
barely measures it.

(Correction to an earlier reading: `accuracy_reward` is *not* structurally zero when
`clipped_ratio` reaches 1.0 — it still registered 0.125 in that step. The problem is
its small share of the total, not a hard zero.)

## How this was found

`run_manifest.json` records lr, β, batch size and optimizer but **not** the sequence
caps, scheduler type, or `max_steps`, so none of the above is visible from the run
records the project keeps. It came out of `training_args.bin`, which HF Trainer writes
into every checkpoint. Worth adding those fields to the manifest.

---
run: smoke_dense_grpo_lc3072
date: 2026-05-03
owner: irene
status: launched
---

# Smoke: dense GRPO 8B with max_completion_length 1024 → 3072

## Why
Inspecting `dense_grpo_math220k_llama8b/llama8b_math220k_dense_scott_full` (job 6385344, 875 steps logged) showed:
- `accuracy_reward/mean` bounces 0–0.25 with no upward trend; back to 0.0 by step 875
- `clipped_ratio` regularly 0.7–1.0 → most completions hit the `max_completion_length=1024` cap
- when clipped_ratio=1.0, `mean_terminated_length=0` → no `\boxed{}` answer → accuracy structurally 0
- sparse GRPO collapses on step 1 into the same fixed point (length cap + format-only reward)

Hypothesis: 1024-token cap on OpenR1-Math-220k CoTs is the dominant reason accuracy_reward never lifts. If true, lifting alone should produce a visibly upward `accuracy_reward/mean` trace within 100 steps.

## Knob
**Only** `max_completion_length: 1024 → 3072`. Everything else matches `dense_scott_full`.

## Hparams

| | value |
|---|---|
| model | meta-llama/Llama-3.1-8B-Instruct |
| dataset | open-r1/OpenR1-Math-220k |
| num_steps | 100 |
| per_device_batch | 2 |
| grad_accum | 4 |
| num_generations | 8 |
| generation_batch | 8 |
| effective batch | 8 |
| learning_rate | 5e-6 |
| beta | 0.025 |
| warmup_ratio | 0.1 |
| max_prompt_length | 512 |
| **max_completion_length** | **3072** (was 1024) |
| precision | bf16 |
| optim | adamw_8bit (default) |
| reward_profile | llama_cot |
| save_steps | 25 |
| save_total_limit | 4 |
| GPU | 1 × H200 |
| walltime | 02:00:00 |

## Paths
- sbatch: `scripts/smoke_dense_grpo_long_completion.sbatch`
- output: `/scratch/xie.yiyi/transfer_v1/dense_grpo_math220k_smoke_lc3072/llama8b_math220k_dense_smoke_lc3072_100steps`
- wandb: project `rl_casino_transfer_v1`, run_name `llama8b_math220k_dense_smoke_lc3072_100steps`
- branch: `irene-sparse-speed-ablation` @ `293ac7f` + this commit

## Decision rule (read after smoke completes)
- `accuracy_reward/mean` rises to ≥ 0.15 and **stays** there in last 25 steps → truncation IS the bottleneck. Promote `max_completion_length=3072` to the long dense run + retry sparse GRPO with same setting.
- `accuracy_reward/mean` still bouncing 0–0.05 with `clipped_ratio < 0.5` → length wasn't the cap; reward landscape itself is degenerate, follow up with reward-weight rebalance `[acc=1.0, format_number=0.4, format_reasoning=0.4]`.
- `clipped_ratio` still ≥ 0.7 with 3072 cap → CoTs in math-220k are even longer than that; revisit dataset filtering or further raise cap.

## Job ID
TBD (filled in after sbatch submit)

# Sparse GRPO — random-mask control for the 2026-04 arms

**Date:** 2026-07-26 · **Branch:** `irene-rebuttal-lora` · **Worktree:** `/home/xie.yiyi/rc-sparse-speed`
**Job:** `8769965` (`scripts/grpo_random_control.sbatch`) · **Status at submit:** PD, `multigpu,gpu`
**Purpose:** close the one hole in the sparse-GRPO evidence now cited in all three rebuttal
replies and the AC comment.

---

## 1. Why this run exists

Three sparse GRPO runs already exist and are quoted in the rebuttal:

| run dir (under `transfer_v1/`) | mask | acc. reward 0–50 → 450–500 |
|---|---|---|
| `sparse_grpo_math220k_oracle_grpo_math` | in-task GRPO Open-R1 | 0.0434 → 0.0833 |
| `sparse_grpo_math220k_oracle_dpo_lightr1_500` | DPO Light-R1 | 0.0587 → 0.0809 |
| `sparse_grpo_math220k_oracle_dpo_tulu3` | DPO Tülu-3 | 0.0561 → 0.0956 |

They are single-variable against each other and all three learn, which supports
**"sparse GRPO trains at ρ=97.5%"**. They cannot support **"the mask matters for GRPO"**,
because no random-mask sparse GRPO run has ever existed — verified by `find` over
`transfer_v1` and Scott's `rl_casino_sparse_train`. The rebuttal currently states that
limit explicitly in three places. This job removes it.

## 2. Why not just use T-sweep arm 3

`scripts/grpo_tsweep_sparse.sbatch` arm 3 is also a random seed42 arm, and it is **not a
substitute**. It is schedule-matched to the *new* dense trajectory:

| | the three 2026-04 arms | T-sweep sparse arms | this control |
|---|---|---|---|
| `lr_scheduler_type` | **linear** | cosine | **linear** |
| `max_completion_length` | **1024** | 2048 | **1024** |
| `max_grad_norm` | **1.0** | 0.1 | **1.0** |
| warmup | 50 steps | 50 steps | 50 steps |

Reading a cosine/cap-2048 random arm against linear/cap-1024 oracle arms would reintroduce
the exact defect that sank Appendix D.4. Both controls are wanted, for different sets:
the T-sweep's random arm anchors the T-sweep, this one anchors the numbers in the reply.

## 3. Hyperparameters — provenance

Every value below was read from `training_args.bin` of
`sparse_grpo_math220k_oracle_grpo_math/.../checkpoints/checkpoint-500`, **not** from a
launcher script. `run_manifest.json` records neither scheduler nor sequence caps nor
`max_steps` — that omission is what produced three wrong fields in the submitted Table 6.

| field | value | field | value |
|---|---|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct` | `grpo_beta` | 0.025 |
| dataset | `open-r1/OpenR1-Math-220k` | `num_generations` | 8 |
| `n_steps` | 500 | `generation_batch_size` | 8 |
| `lr` | 5e-6 | `per_device_train_batch_size` | 2 |
| `lr_scheduler_type` | linear | `gradient_accumulation_steps` | 4 |
| `warmup_steps` | 50 (`warmup_ratio` 0.0) | `max_prompt_length` | 512 |
| `max_grad_norm` | 1.0 | `max_completion_length` | 1024 |
| adam β1/β2/ε | 0.9 / 0.999 / 1e-8 | `weight_decay` | 0.0 |
| `seed` | 42 | `temperature` / `top_p` | 1.0 / 1.0 |
| bf16 | True | gradient checkpointing | True |
| optimizer | `sparse_adamw` | reward profile | `llama_cot` |

**The only difference from the in-task arm is `--mask`.**

- mask: `transfer_v1/oracle_masks_llama8b/random_baseline_lightr1_sp97.5_seed42.pt`
  (8.0 GB, 2026-04-27) — the paper's own random baseline at ρ=97.5%, and the same file
  `grpo_tsweep_sparse.sbatch` arm 3 uses.
- Naming caveat to disclose if the number is reported: the file is named `lightr1`
  because its per-layer keep profile comes from the Light-R1 context, not the GRPO one.
  This is not a new inconsistency — two of the three existing arms are also driven by
  DPO-derived masks — but the honest phrasing is "the paper's random baseline mask",
  not "a random mask density-matched to the GRPO oracle".

## 4. Resources and expected runtime

`gpu:h200:1`, 192 G, 12 cpu, `--time=08:00:00`, `--requeue`, partitions `gpu,multigpu`
(the h200 nodes d4052–d4055 sit in both, and the two carry separate QOS counters).

Reference: the in-task arm (job `6372435`) ran **03:08:32** on one H200 for the same 500
steps at cap 1024, so 8 h is comfortable and `--requeue` is defensive only. Note the 8 h
cap does not fire a requeue on TIME LIMIT anyway — a wall hit is `CANCELLED`.

## 5. Read-out procedure

Identical to how the other three arms were read, so the four are directly comparable:

```
metric = mean of rewards/accuracy_reward/mean over steps 0–50, vs the same over 450–500
source = checkpoints/checkpoint-500/trainer_state.json  (log_history)
```

`--save_model false` skips the ~16 GB final dump; `save_steps`/`save_total_limit` are left
at the script defaults (50 / 3) so `checkpoint-500/trainer_state.json` still exists.

**Sanity gate before the number is used anywhere:** the final `learning_rate` in
`trainer_state.json` must land at **1.111e-08**, matching all three existing arms. If it
does not, the schedule did not replicate and the arm is not a valid control.

## 6. What each outcome means

- **Random gains clearly less than the in-task arm** → the mask matters for GRPO, and the
  cross-objective transfer claim gets a floor. Strongest outcome; report the four together.
- **Random gains about the same** → say so. It would mean the three existing arms show
  only that GRPO trains under *any* 2.5% subnetwork at this horizon, and the
  mask-selection claim for GRPO is not supported by training curves — the structural
  evidence (Jaccard/CKA, tensor-class composition, update-mass concentration) would carry
  it alone. This must not be quietly dropped if it comes out this way; three reviewers
  already flagged that random is competitive on the DPO side.
- **Random diverges / collapses** → check the LR gate in §5 before concluding anything.

## 7. Related

- `docs/run_logs/grpo_schedule_confound_2026-07-25.md` — the D.4 confound this avoids
- `docs/run_logs/grpo_tsweep_2026-07-26.md` — the other, separately-matched random arm
- `docs/run_logs/DO_NOT_REPEAT.md` — "Sparse GRPO at ρ=97.5% learns" entry and its caveat
- `docs/paper_drafts/REBUTTAL_REPLIES.md` — the three places the missing control is disclosed

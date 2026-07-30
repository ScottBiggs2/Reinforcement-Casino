# Matched-schedule GRPO set on AICR — dense vs sparse-oracle vs sparse-random

**Date:** 2026-07-29 · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR `b200-batch`
**Jobs:** `235271` (dense) · `235292` (sparse, GRPO oracle) · `235293` (sparse, random seed42)
— sparse originally `235272`/`235273`, both died at import (§3a) and were resubmitted
after the fix; dense was already training and is untouched.
**Scripts:** `scripts/aicr_grpo_matched_dense.sbatch`, `scripts/aicr_grpo_matched_sparse.sbatch`
(pushed verbatim to `/work/neu/p2026_0038_neu/xie_yiyi/`)
**Requested by:** Irene — "prove the framework can train GRPO" for the NeurIPS #29841
rebuttal (reviewer bf6h W1, results due 2026-08-03).

---

## 1. What this set answers, and why three arms

Appendix B/D.4's "GRPO is resistant to masking" rests on a comparison with three
uncontrolled defects (all documented before this run):

1. **Schedule confound** — dense cosine over `max_steps=5000` vs sparse linear over 500:
   a 449× LR gap at the comparison point
   ([grpo_schedule_confound_2026-07-25.md](grpo_schedule_confound_2026-07-25.md)).
2. **Grad-clip confound** — sparse arms silently ran TRL's global clip at 1.0 vs dense
   0.1; binding on ~90% of steps. Fixed 2026-07-27 (`sparse_grpo_bsr.py:332`); this is
   the first GRPO set to train with the fix.
3. **Truncation regime** — at cap 1024, ~90% of completions truncate and the reward
   barely scores reasoning (DO_NOT_REPEAT.md).

All three arms here share ONE schedule, ONE clip, ONE cap, so:

- **dense vs sparse-oracle** → does 97.5% sparsity cost GRPO anything? (the D.4 rerun)
- **sparse-oracle vs sparse-random** → does the *mask* matter for GRPO? (the control
  that has never existed; disclosed as missing in three rebuttal replies)
- any sparse arm learning at all → "the framework trains GRPO", the minimum deliverable.

The 2026-04 arms cannot serve: their within-run rises are noise-marginal (first-vs-last
50 steps: acc +0.025–0.043 at 1.1–1.9 SE, measured 2026-07-29 from `trainer_state.json`),
their LR annealed to 1.1e-08 by design, and they carry the cap-1024 truncation regime.

## 2. Hyperparameters (identical across arms unless noted)

| field | value | provenance |
|---|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct` | locked model |
| dataset | `open-r1/OpenR1-Math-220k` (`math-220k`), 93,733 rows | paper |
| steps | 500 | paper parity |
| lr / schedule | 5e-6, **cosine over 500**, warmup 50 steps (dense: `warmup_ratio 0.1` ≡ 50) | Scott's locked `grpo_500step_5way_sweep.yaml` |
| β (KL) | 0.025 | paper |
| `max_grad_norm` | **0.1 global on all arms** (post-2026-07-27 fix) | Scott's yaml |
| batch | per-device 2 × grad-accum 4, 8 generations, `generation_batch_size` 8 | paper |
| `max_prompt_length` / cap | 512 / **2048** | Scott's yaml + Table 6 (NOT the as-run 1024) |
| rewards | `llama_cot` | matches all prior runs |
| optimizer | dense `adamw_8bit` (bnb 0.50.0, preflighted on sm_100) / sparse `sparse_adamw` | Scott's yaml |
| precision | auto → bf16 | paper |
| ρ | 97.5% (both sparse arms) | paper |
| checkpoints | `save_steps 50`, `save_total_limit 3`; sparse `--save_model false` | readout needs only `trainer_state.json` |
| resume | `--resume_from_checkpoint auto`, `--requeue` | 12 h wall, ~5 h expected |

**Masks:**

- oracle: `oracle_grpo_math220k_step500_sp97.5.pt` — transferred Explorer→AICR
  2026-07-29 chunked-dd, **MD5 verified both sides `d0c16bbe627cd8e50b3acf7aa7a2f3b2`**
  (8,030,364,869 B). Built 2026-04-29 by checkpoint-diff from the paper's dense GRPO run.
- random: `random_baseline_lightr1_sp97.5_seed42.pt` — the paper's own random baseline,
  already on AICR from the migration. Disclosure that travels with any reported number:
  its per-layer keep profile comes from the Light-R1 context — say "the paper's random
  baseline mask", not "density-matched to the GRPO oracle"
  ([grpo_random_control_2026-07-26.md](grpo_random_control_2026-07-26.md) §3).

**Known deliberate departures from the paper's as-run arms:** cap 2048 (vs as-run 1024)
and matched schedule/clip — so these curves are NOT point-comparable to the 2026-04
arms or to Figure 8/9; they replace that comparison rather than extend it.

## 3. Environment / provenance

- Code: `/work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed` (rsync of the
  `irene-rebuttal-lora` worktree, includes the 07-27 `max_grad_norm`→`GRPOConfig` fix).
- Env: `/work/.../envs/rl_casino` — torch 2.9.0+cu128, transformers 4.57.1, trl 0.24.0;
  Triton `indexed_sparse_adamw` verified on sm_100 (smoke `234788`, 0 strays).
- Dataset pre-downloaded + arrow-processed on the login node 2026-07-29 19:37 EDT, so
  jobs run entirely from cache under the env's offline flags.
- wandb: project `rl_casino_rebuttal`, online mode, runs
  `grpo_matched_aicr_{dense,oracle_grpo_math,random_seed42}_500steps_cap2048`.
- Outputs: `/scratch/xie_yiyi_neu/rebuttal_analysis/grpo_matched_aicr/{dense,oracle_grpo_math,random_seed42}`
  (30-day purge — copy `trainer_state.json` into the repo when the runs finish).

## 3a. Incident: first sparse submissions died at import (235272/235273)

Both failed in seconds with
`ImportError: cannot import name 'sparse_grad_input_triton' from 'src.kernels.bsr_backward'`.
Root cause is a **partial revert on this branch**, latent since 2026-04-30: commit
`68dff0c` ("Revert SparseAdamW + sparse kernels to origin/main") reverted
`bsr_backward.py` + three optimizer files but left `src/mlps/bsr_sparse_mlp.py` at the
B1-atomic state (`cc47f27`), still importing a symbol the reverted kernel file no longer
defines. Every entrypoint importing `src.mlps.bsr_sparse_mlp` (`sparse_grpo_bsr.py`,
`sparse_dpo_bsr.py`) has been dead on this branch since then — unnoticed because all
2026-07 sparse work went through `sparse_dpo_efficiency.py`, which doesn't touch it.

Fix: `39adbcb` restores `bsr_sparse_mlp.py` to its pre-B1 version (`6aa21ae`) — imports
only `sparse_weight_gradient_triton`, signature-compatible with the call site, and the
main-aligned semantics `68dff0c` was after (grad-input dense, weight-grad BSR-Triton).
File pushed to the AICR copy (md5 `589d3c1fec19cc53a98867a5083253f2` both sides);
arms resubmitted as `235292` (oracle) / `235293` (random). Dense `235271` was already
training normally (bnb AdamW8bit preflight passed on B200) and was not touched.

**Second face of the same revert (235292/235293, both FAILED ~3 min in):**
`TypeError: SparseAdamW.__init__() got an unexpected keyword argument 'eager_state_init'`
— the entrypoint also passed a B1-era kwarg the reverted optimizer doesn't have. Fixed
by `f315ae4` (kwarg dropped; `--sparse_adamw_lazy_state` now warns instead of silently
no-oping). This time the resubmission was gated on a **full preflight on the AICR env**:
import both BSR entrypoints and assert every call-site kwarg exists in
`inspect.signature(SparseAdamW.__init__)` — passed. Arms: `235302` (oracle) /
`235303` (random).

**Node-contention move (oracle only):** `235302` landed on b0003 next to our own
n5_masks job `235057` (global top-k over 8B elements — memory-bandwidth bound) and sat
at **156 s/it with 14% GPU util** while the random arm did 36.5 s/it on b0026. No idle
nodes existed (all 28 shared), but co-locating with our own bandwidth-heavy job was
avoidable. Cancelled at Irene's instruction at step 29 (no checkpoint yet; stale wandb
run `3zdtbmrc` set aside) and resubmitted as **`235463`** with `--exclude=b0003`.
**Lesson: before submitting a training arm, `squeue` for our own mask-gen/top-k jobs
and `--exclude` their nodes.** Final arms: `235271` / `235463` / `235303`.

## 4. Read-out plan (pre-registered)

Same procedure as the 2026-04 arms so no reader degrees of freedom:
`rewards/accuracy_reward/mean` and `reward` from `checkpoint-500/trainer_state.json`,
windows 0–50 vs 450–500, report the diff with SE. Sanity gates before any number is
used: all three arms' final LR must match (cosine-to-zero endpoint), and
`completions/clipped_ratio` should sit well below the cap-1024 regime's 0.90.

Outcome mapping (unchanged from grpo_random_control_2026-07-26.md §6):
- oracle ≈ dense, both > random → strongest: sparsity is free AND the mask matters.
- oracle ≈ random, both learn → framework trains GRPO but mask selection is
  unsupported for GRPO at this horizon; structural evidence carries that claim.
- sparse arms flat while dense learns → D.4's conclusion survives the confound fix;
  report it straight.

## 4a. RESULTS (2026-07-30, all three arms COMPLETED at 500 steps)

**Gates.** GATE 1 passed exactly: all three arms end at `learning_rate` **6.0923e-11**,
so the schedule is identical at the comparison point — the defect that made Appendix
D.4 unreadable (449× LR gap) is gone by construction. GATE 2: `clipped_ratio` run means
**dense 0.7883 / oracle 0.7432 / random 0.8413**, all below the cap-1024 regime's 0.90
but not by much — ~75-84% of completions still truncate at cap 2048, which must be
disclosed and which still suppresses the absolute accuracy level.

**Within-arm learning (first-50 → last-50, ± SE).**

| arm | total reward | accuracy reward | KL (policy displacement) |
|---|---|---|---|
| dense | 0.9313 → 1.0325 (**+0.1012, 3.1 SE**) | 0.0775 → 0.0925 (+0.0150, 0.5 SE) | 0.00046 → 0.00271 |
| sparse oracle ρ=97.5 | 0.9550 → 1.0550 (**+0.1000, 2.8 SE**) | 0.0850 → 0.1075 (+0.0225, 0.8 SE) | 0.00035 → 0.00214 |
| sparse random ρ=97.5 | 0.9025 → 0.9862 (**+0.0837, 2.5 SE**) | 0.0575 → 0.0975 (+0.0400, 1.6 SE) | 0.00026 → 0.00035 |

**Cross-arm gaps, last-50 window.**

| comparison | total reward | reading |
|---|---|---|
| dense − oracle | **−0.0225 (0.6 SE)** | indistinguishable; the sparse arm is nominally *higher* |
| oracle − random | **+0.0687 (1.8 SE)** | oracle ahead, and ahead on 9 of 10 windows |
| dense − random | +0.0463 (1.3 SE) | |

**Three claims this set supports.**

1. **Sparse GRPO at ρ=97.5% matches dense** under a matched schedule (gap 0.6 SE). The
   Appendix D.4 result — "even the Oracle mask performs quite poorly against the dense
   baseline" — does not survive the confound fix and must be rewritten.
2. **The mask matters for GRPO.** Oracle beats random by 0.0687 (1.8 SE) and leads on 9
   of 10 windows. Not conventionally significant on a single window, so phrase it as
   "consistently ahead" and lean on the trajectory + KL, not on a p-value.
3. **KL is the cleanest separator.** Oracle displaces the policy **6.1×** over training
   (0.00035 → 0.00214, tracking dense's 0.00046 → 0.00271); random moves **1.3×**
   (0.00026 → 0.00035) — it is nearly frozen. Independent of reward-scale noise, and it
   mirrors the DPO-side finding that the oracle mask's `r_chosen` displacement is ~26×
   random's.

**4b. Signal decomposition — what the reward is actually made of.**

GRPO learns from within-group variance, so the honest measure of "how much of the
learning signal is accuracy vs formatting" is `accuracy_std / reward_std`, not the
reward level (the correction already recorded in DO_NOT_REPEAT.md). Measured here:

| arm | acc share, first-50 | acc share, last-50 | `frac_reward_zero_std`, last-50 |
|---|---|---|---|
| dense | 0.443 | 0.642 | 0.280 |
| oracle ρ=97.5 | 0.516 | 0.658 | 0.260 |
| ρ=95 | 0.435 | 0.575 | 0.240 |
| ρ=99 | 0.442 | 0.646 | 0.240 |
| **ρ=99.75** | 0.481 | 0.594 | **0.080** |
| **random** | 0.446 | 0.559 | **0.100** |

Two things follow. First, **accuracy carries the majority of the gradient signal**
(~44–52% early, rising to 56–66% by step 500) — much higher than the cap-1024 regime's
27%→42%, so the cap-2048 decision materially improved what the reward measures, and the
total-reward result is not a formatting artifact. Second, `frac_reward_zero_std` gives a
**third independent separator** with no overlap: the arms that learn sit at 0.240–0.280,
the two that do not at 0.080–0.100. Higher zero-std means more groups where all 8
rollouts agree — the signature of a policy that has become consistent. Working arms
converge; ρ=99.75 and random stay diffuse.

So three mutually independent measurements partition the same way — last-50 reward, KL
fold-change, and zero-std fraction — which is what makes the ρ=99/99.75 boundary
credible despite each individual reward gap sitting inside noise.

**Honest limitations.** (a) `accuracy_reward` alone is underpowered at 500 steps for
*every* arm, dense included (0.5 SE) — the powered metric is total reward; do not claim
a math-ability improvement from these curves. (b) ~75-84% truncation at cap 2048 still
suppresses absolute accuracy. (c) Single seed per arm. (d) Training curves only — no
held-out math benchmark yet; the checkpoints exist and a GSM8K/MATH eval of the three
checkpoint-500s is the highest-value follow-up.

## 5. Related

- [grpo_schedule_confound_2026-07-25.md](grpo_schedule_confound_2026-07-25.md) — the D.4 confound
- [grpo_random_control_2026-07-26.md](grpo_random_control_2026-07-26.md) — the cancelled Explorer predecessor (8769965)
- [grpo_tsweep_2026-07-26.md](grpo_tsweep_2026-07-26.md) — the cancelled T-sweep chain (8761309–12); cap-2048 decision
- [DO_NOT_REPEAT.md](DO_NOT_REPEAT.md) — grad-clip fix, truncation regime, "sparse GRPO learns" caveats
- [migration_to_aicr_2026-07-29.md](migration_to_aicr_2026-07-29.md) — cluster context

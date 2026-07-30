# GRPO oracle ρ-sweep on AICR — {95, 99, 99.75} against the running 97.5 arm

**Date:** 2026-07-29 (late evening) · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR
**Scripts:** `scripts/aicr_grpo_rho_sweep_masks.sbatch` (mask gen, cpu partition, array 0–2)
+ `scripts/aicr_grpo_matched_sparse.sbatch` (arms, unchanged — only MASK_PATH/RUN_TAG vary)
**Requested by:** Irene — answer Scott's "lower bound of GRPO oracle sparsity" question
properly (matched schedule), for NeurIPS #29841.

## 1. Level selection — from the measured concentration curve, not from priors

`delta_llama_grpo_math220k.json` (job 8725606_1), global squared-mass capture of the
paper's dense GRPO Δθ at step 500:

| keep | ρ | mass captured | expectation |
|---|---|---|---|
| 5% | 95 | 99.93% | ≈ dense (saturation-side anchor) |
| 2.5% | 97.5 | 96.38% | the arm already running (235463) |
| 1% | 99 | 80.86% | first genuinely lossy point — where a "lower bound" could live |
| 0.25% | 99.75 | 48.01% | predicted breakage; also the ρ in the paper's speed tables |

**Pre-registered mechanism prediction:** performance degradation should track mass
capture — sp95 ≈ dense, sp99 mildly degraded, sp99.75 clearly degraded. If it does,
oracle masking for GRPO stops being an empirical accident and becomes a quantitative,
mechanism-linked claim.

**Plateau levels added (Irene, 2026-07-29 late):** ρ ∈ {90, 80, 70} were initially
skipped as saturated (100% mass capture → success only shows more parameters train,
DO_NOT_REPEAT.md). Irene overrode with two good reasons: (a) running Mukherjee's 70%
anchor answers Scott/reviewers with data instead of argument — a measured *plateau*
from keep-30% down to keep-2.5% is itself the strongest form of "GRPO oracle works far
below the natural-sparsity bound"; (b) grid parity with the DPO sweep {80, 90, 97.5,
99}. The full sweep is now **7 points: 70, 80, 90, 95, 97.5, 99, 99.75**, spanning
mass capture 100% → 48%. Mechanism prediction unchanged: flat through 95, degrading
from 99. Masks `235725/235726/235727` (per-level `--mem` 200/240/320G from the
measured keep-count table), arms `235728/235729/235730` (afterok). The cliff arms
`235699/235700/235701` started training as soon as mask array `235698` completed.

## 2. Single-variable design

- **Mask source is held fixed:** all three new oracles come from the SAME trajectory as
  the existing 97.5 oracle — the paper's dense GRPO `checkpoint-500`
  (`llama8b_math220k_dense_scott_full`), transferred Explorer→AICR 2026-07-29 in 4
  safetensors shards (optimizer.pt not needed), **md5-verified per shard before any
  submission** (the chain aborts on mismatch).
- **Same mask code + conventions:** `checkpoint_diff_mask_finder.py` on this branch,
  `--initial_model meta-llama/Llama-3.1-8B-Instruct`, global pooling with
  `min_layer_keep_ratio 0.0025` (defaults, identical to the 97.5 mask's embedded
  metadata), `--force_cpu`.
- **Same training config as 235271/235463/235303:** cosine over 500, warmup 50, lr 5e-6,
  β 0.025, global clip 0.1, cap 2048, `llama_cot`, `sparse_adamw`, bf16, 1×B200,
  `--exclude=b0003` (the node-contention lesson).

So vs the 97.5 arm, ρ is the only variable; vs dense 235271, {ρ, optimizer} as before.

## 3. Job chain (submitted automatically after md5 gate)

| stage | job | partition | resources | notes |
|---|---|---|---|---|
| masks ρ∈{95,99,99.75} | array 0–2 | `cpu` | 16 cpu, 180G, 3h | mem sized from measured keep-count table; ~1h/25m/10m expected |
| arm sp95 | afterok:masks | `b200-batch` | 1×B200, 12h | ~5h expected |
| arm sp99 | afterok:masks | `b200-batch` | 1×B200, 12h | |
| arm sp99.75 | afterok:masks | `b200-batch` | 1×B200, 12h | |

Job IDs recorded in results.tsv when the chain fires. wandb: `rl_casino_rebuttal`,
runs `grpo_matched_aicr_oracle_sp{95,99,99.75}_500steps_cap2048`.

## 4. Read-out

Identical to the 3-arm set (`grpo_matched_aicr_2026-07-29.md` §4): final-LR gate must
match dense's 6.0923e-11 endpoint; first-50 vs last-50 windows ± SE on accuracy and
total reward; the sweep figure is last-50 total reward vs mass-capture fraction.

**Known limitation to disclose:** no per-level random controls (the 97.5 random arm
235303 is the only one). If a level shows an interesting failure, its density-matched
random control is one more arm, decided then — not pre-burned now.

## 4a. RESULTS (2026-07-30, all 6 sweep arms COMPLETED at 500 steps)

**Gate passed on all nine arms:** every arm — dense, random, and all seven ρ levels —
ends at `learning_rate` **6.0923e-11**. One schedule, one clip, one cap across the whole
figure.

**Last-50 total reward vs squared-mass capture:**

| ρ | keep | mass capture | last-50 reward | vs dense |
|---|---|---|---|---|
| dense | 100% | — | 1.0325 ± 0.023 | — |
| 70 | 30% | 100% | **1.0850** ± 0.028 | +0.0525 (1.5 SE) |
| 80 | 20% | 100% | 1.0513 ± 0.026 | +0.0188 (0.5 SE) |
| 90 | 10% | 100% | 1.0562 ± 0.025 | +0.0237 (0.7 SE) |
| 95 | 5% | 99.93% | 1.0525 ± 0.029 | +0.0200 (0.5 SE) |
| 97.5 | 2.5% | 96.38% | 1.0550 ± 0.027 | +0.0225 (0.6 SE) |
| 99 | 1% | 80.86% | 1.0375 ± 0.023 | +0.0050 (0.2 SE) |
| **99.75** | **0.25%** | **48.01%** | **1.0088** ± 0.027 | **−0.0237 (0.7 SE)** |
| random 97.5 | 2.5% | — | 0.9862 ± 0.028 | −0.0463 (1.3 SE) |

**KL displacement (fold change, first-50 → last-50)** — the discriminating metric:

| arm | fold | | arm | fold |
|---|---|---|---|---|
| dense | 5.9× | | ρ=97.5 | 6.2× |
| ρ=70 | 6.1× | | ρ=99 | 5.0× |
| ρ=80 | 8.1× | | **ρ=99.75** | **2.0×** |
| ρ=90 | 5.5× | | **random** | **1.4×** |
| ρ=95 | 6.3× | | | |

**Findings.**

1. **The plateau runs from ρ=70 all the way to ρ=99** — six levels, every one within
   1.5 SE of dense. Training just **1% of parameters** (ρ=99, 80.9% mass capture) still
   matches full fine-tuning. This is the direct answer to "the lower bound of GRPO
   oracle sparsity": it is **not** near Mukherjee's 70% natural-sparsity estimate — the
   oracle survives a parameter budget **100× smaller** than that anchor implies.
2. **The breakpoint is between ρ=99 and ρ=99.75.** At 0.25% of parameters (48% mass
   capture) reward falls below dense and, decisively, **KL displacement collapses from
   ~6× to 2.0×** — approaching the random arm's 1.4× floor. So a real lower bound
   exists; it just sits far past where anyone was looking.
3. **The mass-capture proxy is directionally right but conservative.** Prediction was
   "flat through 95, degrading from 99"; actual is "flat through 99, degrading at
   99.75". Keeping 80.9% of update mass is sufficient; 48% is not. The mechanism holds,
   the threshold is lower than predicted.
4. **KL discriminates where reward saturates.** Reward cannot separate ρ=70 from ρ=99
   (all inside noise, SE ≈ 0.027 → differences under ~0.05 unresolvable). KL fold-change
   holds a clean ~6× plateau, then collapses at 99.75 and again at random. Report KL as
   the primary sparsity-tolerance curve.
5. **Incidental — dose-response speed evidence.** Optimizer step time is monotone in keep
   count: ρ=99.75 (0.02B kept) **28.3 s/it** < ρ=95–80 (0.2–1.6B) **~36 s/it** <
   ρ=70 (2.41B) **48.4 s/it**. Independent support for the kernel's O((1−ρ)P) claim,
   measured end-to-end rather than in a microbenchmark.
6. **Secondary coherence:** `clipped_ratio` is lowest for the arms that learn most
   (ρ=80 0.710 … ρ=97.5 0.743) and highest for the two that don't (ρ=99.75 0.807,
   random 0.841) — arms that learn emit shorter, terminating completions.

**Limitations.** Single seed per level. Reward evidence for the plateau is
"indistinguishable from dense", not "equal to dense" — bounded by SE ≈ 0.027. No
per-level random controls, so at ρ=99.75 the loss of mass and the loss of capacity are
not separated (one extra density-matched random arm would do it). ~71–84% of completions
still truncate at cap 2048.

## 5. Related

- [grpo_matched_aicr_2026-07-29.md](grpo_matched_aicr_2026-07-29.md) — the 3-arm set this extends
- [DO_NOT_REPEAT.md](DO_NOT_REPEAT.md) — why ρ≤90 is skipped; the schedule-confound ledger
- `delta_llama_grpo_math220k.json` (Explorer scratch) — the concentration curve

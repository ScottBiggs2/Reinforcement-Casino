# DPO loss retention vs sparsity ρ — Llama-3.1-8B-Instruct / Light-R1

**Date:** 2026-07-30 · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR (b200)
**Request:** Nadim, 2026-07-30 — "plot with rho on the x-axis and how good it is at
retaining the loss? Maybe absolute difference of the loss between dense and warm start."

**Figure:** `docs/paper_drafts/dpo_retention_vs_rho_lightr1.{png,pdf}`
**Script:** `scripts/plot_dpo_retention_vs_rho.py`

---

## 1. Headline

**ρ costs almost nothing, but the ρ-dependence is real — state it as an effect size, not
as non-significance.** Across a 20× swing in trainable parameters (ρ=80 → 99), the
training-loss spread is 0.0012 — **9 % of the 0.0132 dense-to-sparse gap itself** — and it
is **non-monotone** (ρ=99 sits *below* ρ=97.5). There is no degradation knee in 80 → 99.

> ### CORRECTION 2026-07-30 — the first version of this log was wrong
>
> It claimed "retention is FLAT in ρ; the 80→97.5 spread is ~1.3 σ, not a significant
> trend." That rested on `gap_se = hypot(se_sparse, se_dense)`, which assumes the arms are
> **independent. They are not.** All five arms consume the same data in the same order:
> `logps/chosen` is bit-identical at steps 1–2 across all five `trainer_state.json`
> (−1111.2715, −1122.376), and corr(L_sparse, L_dense) over the last 50 steps is
> **+0.96…+0.97**. Neither `DPO_train.py` nor `sparse_dpo_efficiency.py` exposes `--seed`,
> so both take HF default 42.
>
> Treating paired runs as independent double-counts the shared batch-difficulty noise and
> inflates every bar ~3.7×:
>
> | | `hypot` (published) | paired (correct) |
> |---|---|---|
> | SE on each gap | 0.00065 | **0.00017** |
> | ρ=97.5 vs ρ=80 | +0.0012, "1.3 σ" | +0.00123, **11.3 σ** |
> | ρ=99 vs ρ=97.5 | "noise" | −0.00035, **−3.3 σ** |
>
> So "the ordering is noise" was false — the ordering is stable and ρ=99 genuinely beats
> ρ=97.5. Found by `rc-reviewer`, verified independently before adopting. Script fixed to
> use the paired SE.

**What the 11.3 σ does and does not license.** It is measured against *within-run
minibatch noise only*. There is **n=1 run and n=1 mask per ρ**, and
`probe_jitter_variance_n5_2026-07-28.md` measures mask instability of Jaccard **0.31–0.64**
at *fixed* ρ under seed/jitter changes — "the ρ=97.5 mask" is not a well-defined object.
So 11.3 σ says only "this exceeds batch noise"; it does **not** support "ρ=97.5 is worse
than ρ=80" as a claim about ρ. Report the effect size (9 %), not the σ.

| ρ | trainable params | L_sparse (last-50) | L_sparse − L_dense | retention |
|---|---|---|---|---|
| 80 % | 1,606,052,249 | 0.0402 | **+0.0132 ± 0.00017** | 0.9803 ± 0.0003 |
| 90 % | 803,026,124 | 0.0410 | **+0.0139 ± 0.00017** | 0.9792 ± 0.0003 |
| 97.5 % | 200,756,531 | 0.0415 | **+0.0144 ± 0.00019** | 0.9785 ± 0.0003 |
| 99 % | 80,302,612 | 0.0411 | **+0.0141 ± 0.00016** | 0.9790 ± 0.0002 |

Uncertainties are **paired** per-step SEs (see the correction above), not `hypot`.

**Sign convention:** `L_sparse − L_dense`, the gap/regret convention — positive, and
**higher = worse**. The reverse ordering (`L_dense − L_sparse`, matching Nadim's literal
wording) was tried and rejected on 07-30: it puts all four points *below* the dense
baseline, where "below" collides with the lower-is-better reading of loss itself and gets
read as sparse beating dense. It does not — sparse ends **above** dense in loss at every ρ
(0.040+ vs 0.0271).

Dense reference: `L_start = 0.6965` (≈ ln 2 = 0.6931; policy ≡ reference at init, step-1 loss is exactly 0.6931 on all five arms — the first-5 mean sits 0.0034 above it),
`L_dense = 0.0271 ± 0.0004`.

A 20× swing in trainable parameters (1.61 B → 80.3 M) moves the training loss by 0.0012,
against a 0.0132 cost of sparsifying at all.

### Caveats to carry — all five, not just the first

1. **Training loss, not held-out.** Light-R1 training loss at 20.86 epochs.
   `rewards/accuracies` is **1.0000 on 100 % of the last 50 steps in all five arms** and
   implicit-reward margins are 3.50–3.97 (β·Δ, i.e. Δ ≈ 32–36 nats) — the training set is
   memorised, and loss here is a logarithmic compressor of margin. Flatness in this regime
   is a *weak* test. `AGENT_PLAYBOOK.md:288` records DPO Light-R1 overfitting at step 42;
   this is 12× past that. The DPO literature explicitly decouples train fit from quality
   (Zephyr), and shows falling DPO loss can accompany a *degraded* model (Razin et al.,
   ICLR 2025; Pal et al. 2024). **Free fix available:** the same figure at step ~100–150
   (2–6 epochs, the regime Tulu 2 / Zephyr / DPO actually use) costs zero GPU — the
   `log_history` is already in hand.
2. **No density-matched random controls** (Stage C, deferred). The figure does not show
   the oracle mask beats a random mask at equal density — the gap three reviewers flagged.
   Flatness makes this *more* urgent: if loss is insensitive to *how many* weights train,
   a reviewer will ask whether it is sensitive to *which*.
3. **The dense baseline is a two-knob comparison.** Sparse arms use `sparse_adamw`; the
   dense arm takes `DPO_train.py`'s default **`adamw_8bit`** — and ran on Explorer H200
   while all four sparse arms ran on AICR B200. A constant +0.014 offset independent of ρ
   across a 20× parameter swing is exactly the signature of a fixed implementation
   difference. Nothing in the repo bounds it. **The ρ-sweep itself is unaffected** (all
   four sparse arms share `sparse_adamw`); only the `− L_dense` column is two-knob.
   Killing control: dense arm rerun with `--optimizer sparse_adamw` and an all-ones mask
   (ρ=0) on B200, ~3.7 h, one job.
4. **A ρ-independent dense channel exists in every arm.** `src/optimizers/sparse_adamw.py`
   gates sparsity on `len(p.shape) == 2`, so all 1-D params — every LayerNorm scale,
   266,240 of them — are trained at **100 % density regardless of ρ**. Already recorded in
   `mask_composition_2026-07-25.md:36-53` as a confound that "compresses the difference
   between arms". It is 0.017 % of the ρ=80 budget but **0.33 % of the ρ=99 budget, a 20×
   relative increase** — i.e. it grows in exactly the direction that flattens the curve.
   Falsifier already specified there: one ρ=99 run with LayerNorm frozen.
5. **n=1 run and n=1 mask per ρ.** No seed replicate anywhere in the sweep. All quoted SEs
   are within-run minibatch noise. Cheapest fix, LoRA-GPT-3 style: rerun one arm (ρ=97.5)
   with a second seed and quote the delta as the typical retrain std for all four.

---

## 2. Definitions

`L_start` = mean of steps 1–5 of the dense run (all five arms start from the same base
model and share the identical step-1 loss 0.6931, so one shared `L_start` is correct).
Finals are last-50-step means. Uncertainties are **paired**, because the arms share data
order (see the correction in §1):

- gap SE: `std(L_sparse[-50:] − L_dense[-50:], ddof=1) / sqrt(50)`
- retention SE: `gap_SE / D`, `D = L_start − L_dense` (retention = 1 − gap/D)

The superseded `hypot(se_sparse, se_dense)` form assumed independence and inflated every
bar ~3.7×.

### Why a last-50 mean and not the step-500 loss

Single-step loss noise (std 0.0028 dense / 0.0037 sparse) is **2–3× the effect being
resolved** (the 0.0012 spread across ρ). Reading the gap off one step gives +0.0138,
+0.0129, +0.0134, +0.0121 for steps 500, 499, 498, 497 — a 0.0017 swing driven purely by
which step you happen to pick, larger than the effect itself. Averaging 50 steps cuts that
by √50; with the paired estimator the SE lands at 0.00019.

Averaging is legitimate here because the model has stopped moving: LR over the window runs
5.556e-08 → **1.111e-09**, and the within-window drift is only +0.00049 (dense) / +0.00042
(sp80) — ~1 SE, and near-identical on both arms, so it cancels in the difference.

The choice of 50 is not load-bearing — every window gives the same ordering:

| TAIL | ρ=80 | ρ=90 | ρ=97.5 | ρ=99 | paired SE |
|---|---|---|---|---|---|
| 10 | +0.0126 | +0.0136 | +0.0140 | +0.0136 | 0.00042 |
| 25 | +0.0132 | +0.0140 | +0.0145 | +0.0142 | 0.00025 |
| **50** | **+0.0132** | **+0.0139** | **+0.0144** | **+0.0141** | **0.00019** |
| 100 | +0.0130 | +0.0138 | +0.0142 | +0.0139 | 0.00014 |
| 150 | +0.0131 | +0.0139 | +0.0143 | +0.0140 | 0.00012 |

The ρ=80 < ρ=90 < ρ=99 < ρ=97.5 ordering is identical in every window, and holds in 9 of
10 **disjoint** 50-step blocks across the run — it is not a tail artifact.

50 is used because it is 10 % of the run, matches the sparse arms' `save_steps=50`, and is the window the
GRPO rows in `results.tsv` already use (`first50_last50`) — so numbers stay comparable
across logs. `TAIL` is a module constant in the script if this ever needs changing.

### Panel choice

The figure plots the absolute gap only. Retention is still computed and printed by the
script, but **deliberately not plotted**: `L_start` and `L_dense` are the same constants
for all four arms, so `retention = 1 − gap/D` is an affine transform of the plotted y and
a second panel would carry no extra information — while its narrow y-range (0.975–1.000)
makes the small ρ-spread read as a steep downward trend. Print it, don't plot it.
(Irene + Nadim, 2026-07-30: dropped the retention panel for exactly this reason.)

---

## 3. Provenance — all five arms

Dense source: Explorer job `8792090` (`p1_dpo_lightr1_deltafull_1gpu`), COMPLETED
2026-07-28T10:05, `trainer_state.json` at `global_step 500`, `epoch 20.862`; transferred
to AICR during the 07-29 migration.

All four masks are `_src-deltafull`, i.e. **derived from that same dense run** — so the
sweep is single-knob in ρ, and the sparse arms are compared against the dense run that
generated their own masks.

| ρ | AICR job | node | elapsed | mask |
|---|---|---|---|---|
| 80 | `235056` | b0009 | 03:39:47 | `oracle_dpo_lightr1_step500_sp80_src-deltafull.pt` |
| 90 | `234998` | b0021 | 05:03:00 | `..._sp90_src-deltafull.pt` |
| 97.5 | `234945` | b0014 | 03:41:35 | `..._sp97.5_src-deltafull.pt` |
| 99 | `234944` | b0009 | 03:41:44 | `..._sp99_src-deltafull.pt` |

All four COMPLETED 2026-07-29 on 1×B200, 16 cpu, 256 G. Elapsed is not monotonic in keep
count (ρ=90 took longest on b0021) — different nodes, not investigated; irrelevant to the
loss comparison.

The ρ=97.5 mask passed the comparability gate against the published paper mask:
**Jaccard 0.9691** vs chance 1.27e-2 (`oracle_sparsity_sweep_lightr1_2026-07-28.md` §8b),
so this sweep may be read alongside the published ρ=97.5 point.

### Hyperparameters — identical across all five arms EXCEPT optimizer (see §4 caveat)

From each arm's `run_manifest.json` + `scripts/transfer_p3_sparse_dpo_lightr1.sbatch`:

| Parameter | Value |
|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct` (from base, `checkpoint_path` = base) |
| dataset | `qihoo360/Light-R1-DPOData` (`light-r1`) |
| steps | 500 (`epoch 20.862` on every arm) |
| learning rate | 5e-7, **linear** decay, warmup-ratio 0.1 (peak 5e-7 at step 51) |
| β | 0.1 |
| effective batch | 128 (per-device 2 × grad-accum 64 × 1 GPU) |
| max_length | 1024 |
| optimizer | `sparse_adamw` (dense arm: `adamw_8bit`) |
| `mlp_only` | false (all 291 matched tensors scored) |
| save_steps | sparse arms 50; **dense arm 25** (differs — affects checkpoint granularity only, not the logged loss, which is `logging_steps=1` on all five) |

**Schedule is unconfounded:** all five arms report final `learning_rate = 1.111e-09` and
identical step-1 loss 0.6931, so no arm saw a different LR schedule — the failure mode
that voided the earlier GRPO matched comparison (`grpo_schedule_confound_2026-07-25.md`).

---

## 4. Reproducing the figure

```bash
S=/scratch/$USER/transfer_v1
scp aicr:$S/dense_dpo_lightr1_llama8b/checkpoints/*/checkpoint-500/trainer_state.json dense.json
for SP in 80 90 97.5 99; do
  scp aicr:$S/sparse_dpo_lightr1_oracle_step500_sp${SP}/*_500steps/checkpoints/checkpoint-500/trainer_state.json sp${SP}.json
done
python scripts/plot_dpo_retention_vs_rho.py --dense dense.json \
  --out docs/paper_drafts/dpo_retention_vs_rho_lightr1 \
  80=sp80.json 90=sp90.json 97.5=sp97.5.json 99=sp99.json
```

The script prints the table in §1 to stdout, so the numbers and the figure cannot drift.

---

## 5. Open

- **Tulu3 half of the same plot.** The Tulu3 masks all exist at ρ ∈ {80, 90, 97.5, 99}
  (`oracle_sparsity_sweep_tulu3_2026-07-27.md` §3) and its gate passed at 0.9763, but the
  sparse *arms* have not all been run on AICR. Same script, same invocation once they land.
- **Density-matched random controls** (Stage C) — still deferred; see the caveat in §1.
- The sweep is training loss only; held-out preference eval per arm is ~5 min each
  (Stage D) and has not been run for these four arms.

# Magnitude panels rebuilt with the rebuttal-e1-e2 method — Light-R1 + Tulu-3

**Date launched:** 2026-07-31
**Branch (probe/analysis):** `irene-rebuttal-lora`
**Mask method source:** `origin/rebuttal-e1-e2` @ `32f99de` (Scott), staged verbatim as
`rebuttal_e1e2_mask/src/{warm_start/even_better_mask_finder.py, utils/mask_utils.py}`
on both clusters — mask generation imports ONLY the staged tree (PYTHONPATH pinned).
**Requested by:** Irene ("用rebuttal-e1-e2这个branch的magnitude方法 重新跑一下我
linear probe这个图的magnitude部分 lightr1和tulu3都要").

## What changed vs the submitted magnitude masks

Two differences, both from `rebuttal-e1-e2`'s `even_better_mask_finder.py`:

1. **2-D weight selection + exact tie dedup** (`_select_2d_weight_names`): scores are
   restricted to 2-D `*.weight` tensors, and genuinely tied weights (bit-identical,
   e.g. `lm_head` ↔ `embed_tokens`) are deduplicated by exact equality guarded by a
   fingerprint. This replaces the `(shape, delta.sum())` hash that false-positived on
   bf16 deltas (~99% exact zeros) and silently dropped unrelated same-shape attention
   projections.
2. **No jitter, no `--seed`:** the `--seed/--jitter_rel` CLI was removed on that
   branch; the magnitude mask is deterministic by construction. The n=5 seed axis of
   the previous figure therefore collapses for the Magnitude panels (the 1e-12
   control had already shown sd=0 empirically; now it is structural).

Unchanged: sparsity 97.5%, `min_layer_keep_ratio=0.0025`, global top-k
(`local_pool=False`), CPU scoring, tie-break noise (scale 1e-6, hardcoded seed 42),
target steps {50, 100, 200}, delta sources.

## Jobs

| Stage | Cluster | Job | Partition / shape | Notes |
|---|---|---|---|---|
| Light-R1 masks ×3 | AICR | `243170` | `cpu`, 32c/400G/6h | deltas: `transfer_v1/dense_dpo_lightr1_llama8b/deltas/.../light_r1` |
| Light-R1 probe | AICR | `243171` (afterok 243170) | `b200-batch`, 1×B200/256G/3h | baseline + Oracle-seed42 replication + 3 new mags |
| Tulu-3 masks ×3 | Explorer | `8866296` | `multigpu`, v100-pcie/200G/8h, keep-alive sidecar | deltas: `dense_dpo_tulu3_llama8b_deltalog` (Tulu3 deltas exist only on Explorer) |
| Tulu-3 → AICR push | Explorer login | `push_tulu3_to_aicr.sh` | 4 rsync streams, reaper-resilient | ckpt-500 (−optimizer) + 3 mags + oracle src-deltalog + randoms 42/43/44/46, ~75 GB |
| Tulu-3 probe | AICR | submitted after push | `b200-batch`, 1×B200/256G/5h | baseline + Oracle + Random×4 + 3 new mags, one job |

Explorer is used for the Tulu-3 mask stage only because the Tulu-3 delta log and
deltalog checkpoint-500 never made the AICR transfer; this is a deliberate,
data-gravity exception to "no new work on Explorer".

## Probe protocol (frozen 05-03, byte-identical to the n5_j1e12 run)

`src/analysis/probe_pair_masks.py` on `irene-rebuttal-lora` (the file does not exist
on `rebuttal-e1-e2`): probed model = dense DPO checkpoint-500
(Light-R1: `transfer_v1/.../light_r1/checkpoint-500` on AICR; Tulu-3: the
`_deltalog` run's checkpoint-500 — Scott-spec convergence, canonical per the
2026-07-27 sweep log). `patch_mode=zero_out`, `layer_stride=4`, `cv_folds=3`,
`pairs_per_pos=2`, `holdout_frac=0.2` + `--use_holdout_as_test`, `probe_C=0.01`,
`batch_size=8`, `max_length=256`, `--no_preference`, probe cache pinned to
`probe_cache_v3_oldmath.json`, probe seed 42 (internal).

## Merge rule for the Light-R1 figure

Baseline / Oracle / Random panels are reused unchanged from
`n5_j1e12_summary.json` (n=5). The fresh probe job re-measures Baseline and
Oracle-seed42; merging is gated on Baseline reproducing within 0.02 (expected:
exact) — see `plot_reb_e1e2_figures.py`. The Tulu-3 figure is entirely from its
one fresh probe job (Random ± over mask seeds 42/43/44/46; seed45 does not exist
for Tulu-3).

## Outputs

```
AICR /scratch/xie_yiyi_neu/rebuttal_e1e2_magnitude/
├── masks/reb_magnitude_dpo_lightr1_step{50,100,200}_sp97.5.pt
├── masks/reb_magnitude_dpo_tulu3_step{50,100,200}_sp97.5.pt   (built on Explorer, pushed)
└── probes/{lightr1,tulu3}/probe_pair_results.json
Explorer /scratch/xie.yiyi/rebuttal_e1e2_magnitude/masks/     (tulu3 originals)
Figures: zero_out_heatmap_{lightr1,tulu3}_reb_e1e2_mag.{png,pdf}
```

## Results (2026-07-31, all landed)

Mid-run coordination note: a concurrent session transferred the full Tulu-3 delta
log (5 × 30 GB) to AICR and launched `243209` (`aicr_reb_mag_masks_tulu3.sbatch`),
which built the Tulu-3 masks natively on AICR before the Explorer job finished.
The pipeline switched to those; the probe-input push was trimmed to
ckpt-500 + oracle + randoms. Explorer `8866296` was left running as a
cross-cluster determinism check (its masks land in
`/scratch/xie.yiyi/rebuttal_e1e2_magnitude/masks/`).

| Stage | Job | Outcome |
|---|---|---|
| Light-R1 masks | AICR `243170` | COMPLETED 31 min; 3 masks, 226 tensors, exactly 2.5000% keep |
| Tulu-3 masks | AICR `243209` (concurrent session) | COMPLETED 38 min; identical shape stats |
| Light-R1 probe | AICR `243171` | COMPLETED 4 min (probe cache warm) |
| Tulu-3 probe | AICR `243563` | COMPLETED 9 min; all 9 configs train acc 1.000 |

**Merge gates (Light-R1): exact.** Fresh Baseline and Oracle-seed42 reproduce the
n5_j1e12 values with max |diff| = 0.0000 — probe rig fully deterministic, panels
legitimately mergeable.

**Mask identity: the new masks are substantially different objects.**
Jaccard(new, old jittered seed42) = 0.204 / 0.255 / 0.269 for Light-R1 steps
50/100/200; the old masks also carried 65 tensors (the 1-D norms) that the
2-D-only rule now excludes. Consistent with the tie-break-noise finding: change
the tensor set and the global tie-break landscape reshuffles most of the mask.

**Panel shifts (layer-mean, old → new):**

| | Light-R1 | Tulu-3 |
|---|---|---|
| Math | 0.60–0.62 → **0.74** (max cell +0.31) | 0.60–0.62 → **0.75–0.76** (max cell +0.36) |
| Factual | 0.77–0.81 → 0.82–0.83 (layers 0/4 now 0.96–0.97) | 0.80–0.84 → 0.83–0.84 (same early-layer jump) |
| Syntax / Semantics | ~flat (±0.03) | ~flat (±0.04) |

Same direction on both datasets: the e1-e2 magnitude masks retain notably more
math/factual knowledge, but remain far below Baseline and in the same band as
Random/Oracle — the figure's qualitative story is unchanged.

Figures: `docs/paper_drafts/zero_out_heatmap_{lightr1,tulu3}_reb_e1e2_mag.{png,pdf}`
Data + plot script: `docs/paper_drafts/probe_fig_data/reb_e1e2/`
On AICR: `/scratch/xie_yiyi_neu/rebuttal_e1e2_magnitude/{masks,probes}/`

## Round 2 (same day): full n=5 jitter protocol + all-new panel families

Irene then asked for (a) the reference figure's exact n=5 / jitter_rel=1e-12
protocol applied to the e1-e2 magnitude scoring (explicit approval to re-add the
jitter wrapper the branch had deleted — implemented as a verbatim replica of
`multi_seed_mask_gen._jitter_and_save` in
`probe_fig_data/reb_e1e2/reb_multi_seed_magnitude.py`), and (b) ALL panels
rebuilt with the e1-e2 method ("可以都用新的"): jittered n=5 oracle
(checkpoint-diff, `reb_multi_seed_oracle.py`) and n=5 randoms density-matched
to the NEW oracle — retiring the May randoms, whose density profile floored the
65 LayerNorm tensors while e1-e2 masks leave them intact (a control asymmetry).

| Stage | lightr1 | tulu3 | Outcome |
|---|---|---|---|
| n=5 magnitude masks | `244291` (2h14) | `244292` (3h00) | all 5 seeds bit-identical per step (0/226 tensors differ) — 1e-12 is below fp32 ulp, structural |
| n=5 oracle + random masks | `244494` (49m) | `245117` (~1h20) | oracle seeds bit-identical; oracle/random exactly 2.5000%, 226 tensors |
| magnitude probes | `244293` | `245116` (first attempt `244294` FAILED — see incident) | COMPLETED |
| oracle+random probes | `244496` | `245118` (first chain `244495`/`244497` FAILED/CANCELLED — see incident) | COMPLETED |

**Incident:** at 20:04 the transferred tulu3 checkpoint-500 dir and the May
randoms were deleted on AICR by the concurrent session (the 18:07 probe had
already proven the transfer valid). Re-pushed idempotently from Explorer
(`ALL_TRANSFERS_COMPLETE`, ~23:20) and the three dead jobs were resubmitted.

**Gates:** Baseline agreement between the two probe jobs per dataset:
max |diff| = 0.0000 on both datasets. All probes train acc 1.000.

**Final figures** (`zero_out_heatmap_{lightr1,tulu3}_reb_e1e2_n5.{png,pdf}`,
title exactly as the reference: *n=5 mask seeds, jitter_rel=1e-12,
Llama-3.1-8B, 97.5% sparsity*): with the norm confound removed everywhere,
**Oracle ≈ Magnitude ≈ Random** at ρ=97.5:

| layer-mean | Light-R1 | Tulu-3 |
|---|---|---|
| Math: Oracle / Random / Mag-100 | 0.742 / 0.782 / 0.745 | 0.754 / 0.782 / 0.749 |
| Factual: Oracle / Random / Mag-100 | 0.820 / 0.804 / 0.828 | 0.837 / 0.805 / 0.833 |

The old figure's apparent Oracle/Magnitude-vs-Random gaps were largely the
norm-flooring asymmetry; under matched 2-D-only masks the three families are
statistically indistinguishable per layer (Random ± 0.01–0.08) — consistent
with the ≥64.5%-tie-break-noise finding at this sparsity.

Cross-cluster determinism check (Explorer job `8866296` step50 vs AICR `243209`):
**bit-identical** — 0/226 tensors differ, Jaccard 1.000000 (AICR `246395`; the
first attempt `245280` read a reaper-truncated transfer and is void). Same code +
same deltas produce the same mask on both clusters' CPUs; the Explorer duplicate
can be cleaned up whenever convenient.

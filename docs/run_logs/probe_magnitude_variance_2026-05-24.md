# Probe magnitude levels — seed-variance extension (n=3 mask seeds)

**Date launched:** 2026-05-24
**Branch:** `irene-sparse-speed-ablation`
**Author:** Irene
**Extends:** `probe_magnitude_levels.sbatch` (commit `023f89f`, May 6 v3)
**Sbatch:** `scripts/probe_magnitude_variance.sbatch`
**Aggregation:** `src/analysis/probe_variance_plots.py --absolute_heatmap_label ...`

## Goal
Add ±std error annotations to the May 6 v3 zero-out heatmaps by rerunning
the pairwise linear probes for n=3 mask-construction seeds (42, 43, 44).
Same DPO checkpoints, same probe pipeline — only the mask seed varies.

## Variance source (literal interpretation, by request)
| Mask | Stochastic across mask seeds? | Expected std on chart |
|---|---|---|
| Oracle | No (`|θ_500 − θ_0|` checkpoint diff) | 0.000 |
| Magnitude-50/100/200 | No (top-k by accumulated delta) | 0.000 |
| Random-seedX | Yes (`torch.manual_seed(X)` in `random_mask_baseline.py`) | non-zero |

The probe itself (`train_pairwise_probes`) uses a fixed implicit seed=42
inside `probe_pair_masks.py`, so probe-level randomness does NOT contribute.
This is intentional: variance reflects mask construction only.

## Configs (per dataset, per seed)
```
[
  Oracle,
  Random-seedX,           # X ∈ {42, 43, 44}
  Magnitude-50,
  Magnitude-100,
  Magnitude-200
]
```
Plus the implicit unmasked `Baseline\n(no mask)` panel that
`probe_pair_masks.py` adds.

## Hyperparameters (frozen from May 6 v3 run)
| Param | Value |
|---|---|
| Backbone | `meta-llama/Llama-3.1-8B-Instruct` |
| Datasets | tulu3, lightr1 (dense DPO ckpt-500) |
| Sparsity | 97.5% |
| `min_layer_keep_ratio` | 0.0025 |
| Probe mode | `zero_out`, `--no_preference` (4 benchmark probes) |
| `layer_stride` | 4 |
| `cv_folds` | 3 |
| `pairs_per_pos` | 2 |
| `holdout_frac` | 0.2 (`--use_holdout_as_test`) |
| `probe_C` | 0.01 |
| `batch_size` | 8, `max_length` 256 |
| Probe seed (internal) | 42 (fixed; see note above) |
| Random-mask seed | 42, 43, 44 |

## Provenance
- **Oracle masks** (deterministic): `/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/oracle_dpo_{tulu3,lightr1}_step500_sp97.5.pt`
- **Magnitude-T masks** (deterministic): `/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b/warm_magnitude_dpo_{tulu3,lightr1}_step{50,100,200}_sp97.5.pt`
- **Random masks (42/43/44)**: `/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/random_baseline_{tulu3,lightr1}_sp97.5_seed{42,43,44}.pt` (already present on cluster)
- **Seed-42 probe results**: reused from May 6 v3 run at `/scratch/xie.yiyi/probe_magnitude_levels/{tulu3,lightr1}/zero_out/probe_pair_results.json`
- **New seed-43/44 probe results**: `/scratch/xie.yiyi/probe_magnitude_variance/{tulu3,lightr1}/seed{43,44}/probe_pair_results.json`

## Outputs
- `/scratch/xie.yiyi/probe_magnitude_variance/summary/{tulu3,lightr1}/`
  - `zero_out_heatmap_{tag}_v4_variance.png` — v3 visual style, RdYlGn,
     cells annotated `mean ± std` across n=3 mask seeds.
  - `summary_delta_heatmap.png`, `summary_bar_chart.png`,
     `probe_pair_delta_all.png` — Δ-from-baseline views.
  - `variance_summary.json` — machine-readable mean/std tables.

## Resource budget
- 1× H200 (multigpu, `--exclude=d1025`)
- `--time=02:30:00`, 8 CPU, 96 GB RAM
- ~3 probe passes × 2 datasets = 6 passes; seed42 is `cp` (free), so 4 fresh
  probe passes total. Each took ~15 min in the May 6 v3 run → ~1h compute.

## Hypotheses
- All deterministic rows (Oracle, Magnitude-T) will show ±0.000.
- Random-seedX will show some spread, especially at lower-accuracy probes.
- If Random's ±std envelope **overlaps** Oracle/Magnitude central values
  (very likely from May 6 absolute-accuracy table), the v3 negative finding
  ("warm-magnitude ≈ random") strengthens: the gap is below mask-seed noise.

## Follow-ups
- If the chart looks good, fold a second figure into
  `docs/paper_drafts/probe_magnitude_levels_report.tex` and refresh Table 1
  with `mean ± std` columns for Random.
- Decide whether to also run probe-seed variance (requires patching
  `probe_pair_masks.py` to expose `--probe_seed`).

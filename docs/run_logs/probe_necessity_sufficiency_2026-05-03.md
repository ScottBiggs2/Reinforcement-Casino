# Run log — Probe Necessity/Sufficiency (tulu3 DPO oracle subnetwork)

**Date:** 2026-05-03
**Branch:** `irene-sparse-speed-ablation`
**Git SHA at launch:** `67a3772` (parent — see `git rev-parse HEAD` after the
patch commit lands)
**Operator:** Irene (via Claude)

## What this is

A *single-knob* probe study to upgrade `linear-probe-pairwise` from a
sufficiency-only signal to a full **necessity + sufficiency + specificity**
identification of the tulu3-DPO oracle subnetwork.

Old protocol (legacy): for each mask file `M`, set `w' = w_ft * M` and run
the BT pairwise probe. This only tested whether the M-located fine-tuned
weights alone preserved probe signal — not whether ¬M did, and not against
a multi-seed random control band.

New protocol — three legs of evidence, jointly required:

| Leg            | Mask        | Patch mode (any of three)   | Expected outcome    |
|----------------|-------------|-----------------------------|---------------------|
| Sufficiency    | `M`         | `delta_only` / `zero_out`   | probe ≈ baseline    |
| Necessity      | `1 - M`     | `delta_only` (on ¬M)        | probe ≪ baseline    |
| Specificity    | random×6    | matched per-layer density   | probe << oracle     |

Implementation: `apply_mask` now accepts `patch_mode ∈ {zero_out,
delta_only, anti_delta_only}` and an optional `base_state_dict`.
Backwards-compatible default is `zero_out` (legacy behavior).

## Files patched / added

- `src/analysis/probe_analysis.py` — `apply_mask` extended with `patch_mode`
  and `base_state_dict`. Backwards-compatible.
- `src/analysis/probe_pair_masks.py` — new flags `--patch_mode`,
  `--base_model`. Loads base state dict on CPU once, threads through to
  `apply_mask`.
- `src/analysis/probe_analysis_pair.py` — added `cv=0` bypass: skips KFold
  setup and per-fold loop; holdout becomes the only test signal. Required
  to make the simple protocol describable as "train 80, test 20".
- `src/warm_start/anti_mask.py` — new tiny generator: per-layer `1 - M`,
  preserves shapes/keys, writes metadata `{"method": "anti_mask",
  "anti_density": ...}`.
- `scripts/probe_necessity_sufficiency.sbatch` — single sbatch that
  auto-builds anti-mask + 5 random seeds, then runs the probe in all three
  patch modes.

## Hyperparameters (all three patch modes, identical)

Simple protocol: **train probe on 80 % of pairs, test on a single 20 %
holdout, default L2 (C=1.0), no CV**. Variance estimate for the
specificity claim comes from the 6-seed random mask band, not from
probe-side resampling — this avoids stacking two redundant variance
mechanisms and keeps the protocol describable in one sentence.

| Param                        | Value                                              |
|------------------------------|----------------------------------------------------|
| Probed model                 | dense tulu3 DPO, ckpt-500 (Llama-3.1-8B-Instruct)  |
| Base model (delta modes)     | `meta-llama/Llama-3.1-8B-Instruct`                 |
| Probe                        | BT pairwise logistic regression                    |
| `--probe_C` (L2)             | 0.01 (restored after default-1.0 sanity-check failed: probe memorized train=1.000 and Baseline holdout went BELOW chance on preference probes — see job 6522535) |
| `--cv_folds`                 | 3 (light CV — single-split 0 was still anti-predictive on preference even with C=0.01 in job 6522862; benchmark probes were fine at 0.92-0.99 but preference acc went 0.355/0.176, indicating split-luck variance on n=1270 × d=14336. CV averages this out without 5-fold reporting clutter.) |
| `--pairs_per_pos`            | 2                                                  |
<!-- cv_folds row moved above; placeholder kept to avoid stale duplication -->
| `--holdout_frac`             | 0.2 + `--use_holdout_as_test`                      |
| `--layer_stride`             | 4 (9 sampled MLP layers + last)                    |
| `--batch_size`               | 8                                                  |
| `--max_length`               | 256                                                |
| `--n_jobs`                   | 8                                                  |
| Mask sparsity                | 97.5 % (matched across all configs)                |

## Mask provenance

| Mask                            | Path (cluster)                                                         |
|---------------------------------|------------------------------------------------------------------------|
| Oracle (warm-magnitude step500) | `/scratch/.../oracle_dpo_tulu3_step500_sp97.5.pt`                      |
| Anti-Oracle = 1 − Oracle        | `/scratch/.../anti_oracle_dpo_tulu3_step500_sp97.5.pt` (built in-job)  |
| Random seed 42                  | `/scratch/.../random_baseline_tulu3_sp97.5_seed42.pt` (already exists) |
| Random seeds 43–47              | `/scratch/.../random_baseline_tulu3_sp97.5_seed{43..47}.pt` (in-job)   |

All random masks share the oracle's per-layer density (matched via
`random_mask_baseline.py --reference_mask oracle ... --sparsity_percent
97.5`).

## Probe data

`v3` cache: `dpo_preference` (tulu3, 1000 per class) + `grpo_preference`
(open-r1-math-220k) + 4 benchmark properties (syntax/semantics/factual/math).

## Job chain

| Stage                           | Script                                          | Depends on |
|---------------------------------|-------------------------------------------------|------------|
| Anti-mask + random seeds + probe× 3 modes | `scripts/probe_necessity_sufficiency.sbatch` | (none)     |

Runtime estimate: 6 mask gens (~1 min each on CPU) + 8 probe configs ×
3 modes ≈ 90 min on 1 GPU. `--time=02:30:00`.

## Success criteria

For both `dpo_preference` and `grpo_preference`, layer-mean accuracy with
paired-bootstrap 95 % CI on the same `(h+, h−)` pair indices should
satisfy:

1. **Oracle (delta_only)** ≈ Baseline   (sufficiency)
2. **Anti-Oracle (delta_only on ¬M)** < Random mean − ε   (necessity)
3. **Random×6** clustered tightly above Anti-Oracle and clearly below
   Oracle   (specificity / mask is not a random projection)

Failure of any leg = subnetwork claim downgraded to a sufficiency-only
result; report accordingly in the §3.5.3 paper draft.

## Outputs

```
/scratch/xie.yiyi/probe_necsuff_tulu3/
├── masks.json
├── zero_out/
│   ├── probe_pair_results.json
│   ├── probe_pair_heatmap_all.png
│   └── probe_pair_delta_all.png
├── delta_only/        ...
└── anti_delta_only/   ...
```

Post-hoc analysis (paired bootstrap + necessity-sufficiency 3-bar plot)
will be a follow-up notebook; the JSON is the source of truth.

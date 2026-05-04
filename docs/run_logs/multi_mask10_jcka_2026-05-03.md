# Run log — Multi-mask (10) Jaccard + CKA

**Date:** 2026-05-03
**Branch:** `irene-sparse-speed-ablation`
**Job ID:** 6522694
**Operator:** Irene (via Claude)

## What this is

Replaces the earlier 4-mask analysis (DPO-LightR1-500, DPO-Tulu3-500,
GRPO-OpenR1-500, Random) with a 10-mask sweep that adds early-step DPO
masks, a second GRPO step, the matched-schema Tulu3 random baseline, and
both warm-magnitude masks (Light-R1 + Tulu3). The pipeline now also runs
Scott-style param-bucket and decoder-layer aggregations and emits matrix
CSVs + heatmaps so the comparison stays readable past pair-count > 10.

## The 10 masks

| # | label | family | dataset | step | sp | path (under `/scratch/xie.yiyi/transfer_v1/`) |
|---|---|---|---|---|---|---|
| 1 | DPO-LightR1-s150-sp97.5 | DPO-oracle | LightR1 | 150 | 97.5 | `oracle_masks_llama8b/oracle_dpo_lightr1_step150_sp97.5.pt` |
| 2 | DPO-LightR1-s500-sp97.5 | DPO-oracle | LightR1 | 500 | 97.5 | `oracle_masks_llama8b/oracle_dpo_lightr1_step500_sp97.5.pt` |
| 3 | DPO-Tulu3-s150-sp97.5 | DPO-oracle | Tulu3 | 150 | 97.5 | `oracle_masks_llama8b/oracle_dpo_tulu3_step150_sp97.5.pt` |
| 4 | DPO-Tulu3-s500-sp97.5 | DPO-oracle | Tulu3 | 500 | 97.5 | `oracle_masks_llama8b/oracle_dpo_tulu3_step500_sp97.5.pt` |
| 5 | GRPO-Math220k-s500-sp97.5 | GRPO-oracle | OpenR1-Math220k | 500 | 97.5 | `oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt` |
| 6 | GRPO-Math220k-s850-sp97.5 | GRPO-oracle | OpenR1-Math220k | 850 | 97.5 | `oracle_masks_llama8b/oracle_grpo_math220k_step850_sp97.5.pt` |
| 7 | Random-LightR1-sp97.5-s42 | Random | LightR1-schema | — | 97.5 | `oracle_masks_llama8b/random_baseline_lightr1_sp97.5_seed42.pt` |
| 8 | Random-Tulu3-sp97.5-s42 | Random | Tulu3-schema | — | 97.5 | `oracle_masks_llama8b/random_baseline_tulu3_sp97.5_seed42.pt` |
| 9 | WarmMag-LightR1-s200-sp97.5 | WarmMag | LightR1 | 200 | 97.5 | `warm_masks_llama8b/warm_magnitude_dpo_lightr1_step200_sp97.5.pt` |
| 10 | WarmMag-Tulu3-s200-sp97.5 | WarmMag | Tulu3 | 200 | 97.5 | `warm_masks_llama8b/warm_magnitude_dpo_tulu3_step200_sp97.5.pt` |

C(10, 2) = **45 pairs**. Mask family counts: DPO-oracle 4, GRPO-oracle 2,
Random 2, WarmMag 2.

## Three priority comparisons (English)

1. **Mask stability across training step** (same algo × same data, different step):
   1↔2, 3↔4, 5↔6 — Jaccard ≈ 1 means the subnetwork is locked in early.
2. **Algorithm vs dataset identity**:
   - same-algo / cross-data: 1↔3, 1↔4, 2↔3, 2↔4 (DPO across LightR1 / Tulu3)
   - cross-algo at the family level: mean Jaccard `{DPO-*}` vs `{GRPO-*}`
   Stress-tests the 0.79 vs 0.13 split observed in the 4-mask study.
3. **Oracle vs WarmMag** (is the expensive ckpt-diff oracle just magnitude?):
   2↔9 (DPO-LightR1-s500 vs WarmMag-LightR1-s200) and 4↔10 (Tulu3 counterpart).
   High Jaccard *and* high CKA → magnitude is a cheap surrogate.
   Low Jaccard but high CKA → different params, same function (subnetwork redundancy).
   Both low → oracle is genuinely picking up something beyond magnitude.

Floor (always reported): each mask vs the matched-schema random
(LightR1-seed42 or Tulu3-seed42).

## Code changes (vs 4-mask version)

- **Cherry-picked from `cav_fixes`** (no edits):
  - `src/cold_start/mask_jaccard_aggregates.py` — `extended_jaccard_report` with
    `by_param_bucket` (attn/mlp/norm/other) + `by_decoder_layer` (0..31 + non_decoder).
  - `src/cold_start/export_layer_metrics_csv.py` — already present locally,
    matches `cav_fixes` byte-for-byte (CSV merger of jaccard.json + cka.json).
- **Rewritten** `scripts/multi_mask_jaccard_cka.py`:
  - new `masks.json` schema with `family / dataset / step / sparsity`
  - per-pair Jaccard now uses `extended_jaccard_report(...)` (param-bucket + decoder-layer)
  - additional outputs (alongside the existing JSON):
    - `jaccard_matrix.csv`, `cka_matrix.csv` (10×10, sortable in any spreadsheet)
    - `family_matrix_jaccard.csv`, `family_matrix_cka.csv` (4×4 family × family means)
    - `pairs_long.csv` (one row per pair: families, datasets, steps, sparsities,
      jaccard global + attn/mlp/norm/other buckets, cka mean/min/max)
    - `jaccard_heatmap.png`, `cka_heatmap.png` (sorted by family, family separator lines)
    - `per_decoder_layer.png` (32 × 45-pair line plot of layer-wise Jaccard)
- **New sbatch** `scripts/multi_mask10_jcka.sbatch`:
  - h200×1, 02:00:00 walltime, requeue
  - inlines the 10-mask manifest as a heredoc, pre-flights every path
  - calls `python scripts/multi_mask_jaccard_cka.py --calibration_dataset tulu3`

## Hyperparameters

| Param | Value |
|---|---|
| Probed model | meta-llama/Llama-3.1-8B-Instruct (bf16) |
| Layer substr | `down_proj` |
| Calibration dataset | `tulu3` (HF id: `allenai/tulu-3-sft-mixture`) |
| Calibration samples | 64 |
| Max length | 512 |
| Batch size | 4 |
| Seed | 42 |
| Sparsity (all masks) | 97.5 % |

## Outputs

- Output dir on cluster: `/scratch/xie.yiyi/transfer_v1/multi_mask10_analysis/`
- Logs: `logs/mm10_jcka_6522694.{out,err}` (cluster worktree)

## Next steps

1. Wait for job 6522694 → inspect `pairs_long.csv` + heatmaps.
2. If the **Oracle vs WarmMag** Jaccard is high, drop the ckpt-diff oracle
   from the §3.5 narrative as a baseline (move it to "checkpointed
   variant of magnitude"); if low, keep the oracle as the headline mask.
3. Use the per-decoder-layer line plot to pick the 3-4 most-divergent
   layers for a follow-up necessity/sufficiency probe (instead of all 32).

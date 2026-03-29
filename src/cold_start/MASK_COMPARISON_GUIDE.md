# Mask Comparison & Charting Suite

End-to-end toolkit for comparing sparse masks **structurally** (Jaccard) and
**functionally** (linear CKA), exporting per-layer metrics to CSV, and
generating publication-ready plots.

---

## Quick start

The fastest way to run the full pipeline is the orchestrator script:

```bash
sbatch scripts/gen_grpo_masks_and_plot.sh
```

This generates SNIP/CAV/Fisher masks for both GRPO and DPO modes, computes
pairwise CKA and Jaccard, exports CSVs, and produces 2x2 diagnostic plots --
all in one job. See [Orchestrator script](#orchestrator-script) for details.

To run individual steps manually, read on.

---

## Tools

### 1. `mask_to_jaccard.py` -- Structural comparison

Measures what fraction of selected parameters are shared between two masks.

```
J(A, B) = |A intersection B| / |A union B|
```

A score of 1.0 means identical masks; 0.0 means zero overlap.

**Usage**

```bash
python src/cold_start/mask_to_jaccard.py MASK_A.pt MASK_B.pt [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `mask_a` | required | First mask file (.pt) |
| `mask_b` | required | Second mask file (.pt) |
| `--output / -o` | auto | Output JSON path (auto: `jaccard_<a>_vs_<b>.json`) |
| `--device` | `cpu` | `cpu` or `cuda` |

**Output JSON**

```json
{
  "mask_a": "/abs/path/a.pt",
  "mask_b": "/abs/path/b.pt",
  "jaccard": {
    "aggregate": 0.0523,
    "mean": 0.0501,
    "min": 0.0312,
    "max": 0.0734,
    "n_layers": 54,
    "total_intersection": 12345,
    "total_union": 236000
  },
  "per_layer_jaccard": {"model.layers.0.mlp.gate_proj.weight": 0.0523, "...": "..."},
  "metadata_a": {"...": "..."},
  "metadata_b": {"...": "..."}
}
```

---

### 2. `mask_to_cka.py` -- Functional comparison

Measures how similar the *activations* of two masked subnetworks are on a
calibration dataset using **linear CKA** (Centered Kernel Alignment). Unlike
Jaccard, CKA captures functional equivalence -- two structurally different
masks can produce similar representations.

**Usage**

```bash
python src/cold_start/mask_to_cka.py MASK_A.pt MASK_B.pt [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `mask_a` | required | First mask file (.pt) |
| `mask_b` | required | Second mask file (.pt) |
| `--model_name` | `google/gemma-3-270m-it` | HuggingFace model |
| `--compare` | `mask_vs_mask` | Comparison mode (see below) |
| `--n_samples` | `64` | Calibration samples |
| `--max_length` | `512` | Max token length |
| `--batch_size` | `4` | Forward-pass batch size |
| `--seed` | `42` | RNG seed |
| `--output / -o` | auto | Output JSON path |
| `--device` | `cpu` | `cpu` or `cuda` |

**Comparison modes**

| Mode | What is compared |
|------|-----------------|
| `mask_vs_mask` | Subnetwork A activations vs subnetwork B activations |
| `original_vs_a` | Full (unmasked) model vs subnetwork A |
| `original_vs_b` | Full (unmasked) model vs subnetwork B |
| `chosen_vs_rejected` | Subnetwork A on chosen texts vs rejected texts |

**Output JSON**

```json
{
  "mask_a": "/abs/path/a.pt",
  "mask_b": "/abs/path/b.pt",
  "model_name": "google/gemma-3-270m-it",
  "compare": "mask_vs_mask",
  "label_a": "mask_a_subnetwork",
  "label_b": "mask_b_subnetwork",
  "n_samples": 64,
  "seed": 42,
  "cka": {
    "mean": 0.9821,
    "min": 0.9102,
    "max": 0.9991,
    "n_layers": 18,
    "n_skipped": 0
  },
  "per_layer_cka": {"model.layers.0.mlp.down_proj": 0.9821, "...": "..."},
  "metadata_a": {"...": "..."},
  "metadata_b": {"...": "..."}
}
```

---

### 3. `export_layer_metrics_csv.py` -- Per-layer CSV export

Combines two mask files (and optional CKA/Jaccard JSON reports) into a single
CSV with per-layer sparsity, effective rank, Jaccard, and CKA columns.

**Usage**

```bash
python src/cold_start/export_layer_metrics_csv.py MASK_A.pt MASK_B.pt [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `mask_a` | required | First mask file (.pt) |
| `mask_b` | required | Second mask file (.pt) |
| `--cka-json` | none | CKA JSON from `mask_to_cka.py` |
| `--jaccard-json` | none | Jaccard JSON from `mask_to_jaccard.py` (auto-computed if omitted) |
| `--output / -o` | auto | Output CSV path |

**Output CSV columns**

```
layer, layer_index, shape_a, shape_b, n_params,
kept_a, density_a, sparsity_a, effective_rank_a, effective_rank_a_norm,
kept_b, density_b, sparsity_b, effective_rank_b, effective_rank_b_norm,
jaccard, cka
```

---

### 4. `convert_json_reports_to_csv.py` -- Batch JSON-to-CSV conversion

Scans a directory for JSON reports (from `mask_to_jaccard.py` or
`mask_to_cka.py`) and batch-converts them into per-layer CSVs plus summary
CSVs.

**Usage**

```bash
python src/cold_start/convert_json_reports_to_csv.py --input-dir masks/ [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `masks` | Directory containing JSON reports |
| `--output-dir` | same as input | Where to write CSVs |
| `--recursive` | off | Scan subdirectories |

**Output** (for each JSON file):
- `{stem}.csv` -- full per-layer metrics (same schema as `export_layer_metrics_csv.py`)
- `{stem}_summary.csv` -- single-row aggregate: CKA mean/min/max, Jaccard aggregate/mean/min/max, source paths

---

### 5. `plot_layer_metrics_csv.py` -- Visualization

Generates 2x2 panel plots from per-layer CSV files.

**Single-file mode** (one plot per CSV in a directory):

```bash
python src/cold_start/plot_layer_metrics_csv.py --input-dir masks/grpo_verify
```

| Panel | Content |
|-------|---------|
| Top-left | Per-layer Jaccard + random baseline |
| Top-right | Per-layer CKA + random baseline |
| Bottom-left | Per-layer sparsity (mask A and B) |
| Bottom-right | Per-layer effective rank, normalized (mask A and B) |

The random Jaccard baseline is computed analytically:
`E[J] = (d_a * d_b) / (d_a + d_b - d_a * d_b)` where `d` = density.

**Comparison mode** (overlay two CSVs):

```bash
python src/cold_start/plot_layer_metrics_csv.py \
    --compare \
    --csv-a masks/grpo_verify/layer_metrics_snip_grpo_vs_dpo.csv \
    --csv-b masks/grpo_verify/layer_metrics_cav_grpo_vs_dpo.csv \
    --label-a "SNIP" \
    --label-b "CAV" \
    --output masks/grpo_verify/compare_SNIP_vs_CAV.png
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--compare` | off | Enable comparison mode |
| `--csv-a` | required | First CSV |
| `--csv-b` | required | Second CSV |
| `--label-a` | `A` | Legend label for first CSV |
| `--label-b` | `B` | Legend label for second CSV |
| `--output / -o` | auto | Output PNG path |
| `--input-dir` | `masks` | Directory for single-file mode |
| `--recursive` | off | Scan subdirectories in single-file mode |
| `--pattern` | `*.csv` | Glob pattern in single-file mode |

Output: 220 DPI PNG files.

---

### 6. `mask_utils.py:compute_jaccard_similarity` -- Programmatic Jaccard

Utility function for computing Jaccard from Python code (not a CLI tool).
Used by `random_mask_baseline.py` and other scripts.

```python
from src.utils.mask_utils import compute_jaccard_similarity

result = compute_jaccard_similarity(pred_masks, true_masks)
# result keys:
#   aggregate_jaccard, mean_jaccard, min_jaccard, max_jaccard,
#   per_layer (dict), overlap_fraction_predicted,
#   overlap_fraction_reference, cosine_similarity
```

Uses CUDA if available. Includes overlap fractions and cosine similarity in
addition to Jaccard.

---

## When to use which tool

| Question | Tool |
|----------|------|
| Do two masks select the same parameters? | `mask_to_jaccard.py` |
| Do two masks produce similar representations? | `mask_to_cka.py` |
| How does a mask change model behavior vs unmasked? | `mask_to_cka.py --compare original_vs_a` |
| Does a mask treat chosen/rejected differently? | `mask_to_cka.py --compare chosen_vs_rejected` |
| Need per-layer stats in a spreadsheet? | `export_layer_metrics_csv.py` |
| Have many JSON reports to convert? | `convert_json_reports_to_csv.py` |
| Need diagnostic plots? | `plot_layer_metrics_csv.py` |

A high Jaccard implies high CKA, but not vice versa -- CKA can reveal
functional agreement even between structurally different masks.

---

## Data flow

```
inference_mask_finder.py
    |
    v  (generates .pt masks)
    |
    +---> mask_to_jaccard.py ---------> jaccard_*.json ---+
    |                                                     |
    +---> mask_to_cka.py -------------> cka_*.json -------+
                                                          |
                                                          v
                                            export_layer_metrics_csv.py
                                                          |
                                                          v
                                                layer_metrics_*.csv
                                                          |
                                                          v
                                            plot_layer_metrics_csv.py
                                                          |
                                                          v
                                                    *_plots.png

Batch alternative:
    masks/*.json ---> convert_json_reports_to_csv.py ---> *.csv + *_summary.csv
```

---

## Orchestrator script

`scripts/gen_grpo_masks_and_plot.sh` runs the full pipeline as a single SLURM
job. It generates 6 masks (SNIP/CAV/Fisher x GRPO/DPO), computes all pairwise
CKA and Jaccard, exports CSVs, plots, and prints an aggregate summary.

**Configurable environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `google/gemma-3-270m-it` | Model name |
| `N_SAMPLES` | `64` | Calibration samples |
| `SPARSITY` | `90.0` | Target sparsity % |
| `MASK_DIR` | `masks/grpo_verify` | Output directory |
| `SKIP_DPO` | `0` | Set to `1` to reuse existing DPO masks |
| `BATCH_SIZE` | `1` | Inference batch size |
| `MAX_LENGTH` | `256` | Max token length |
| `FORCE_GPU` | `1` | Fail if CUDA unavailable |
| `CPU_THREADS` | auto | Limit CPU threads |

**Pipeline steps:**

1. Generate 6 masks via `inference_mask_finder.py`
2. Compute CKA for within-method GRPO-vs-DPO pairs (3 JSON files)
3. Export per-layer CSVs for all pairs (5 CSVs)
4. Generate individual 2x2 plots for each CSV
5. Generate overlay comparison plot (SNIP vs CAV)
6. Print aggregate Jaccard summary table

**Output tree:**

```
masks/grpo_verify/
  snip_grpo.pt, snip_dpo.pt
  cav_grpo.pt, cav_dpo.pt
  fisher_grpo.pt, fisher_dpo.pt
  cka_snip_grpo_vs_dpo.json
  cka_cav_grpo_vs_dpo.json
  cka_fisher_grpo_vs_dpo.json
  layer_metrics_snip_grpo_vs_dpo.csv
  layer_metrics_cav_grpo_vs_dpo.csv
  layer_metrics_fisher_grpo_vs_dpo.csv
  layer_metrics_grpo_snip_vs_cav.csv
  layer_metrics_grpo_snip_vs_fisher.csv
  *_plots.png (one per CSV)
  compare_SNIP_vs_CAV_grpo_dpo.png
```

---

## Mask file format

All tools expect PyTorch `.pt` files with this structure:

```python
{
    "masks": {
        "model.layers.0.mlp.gate_proj.weight": tensor([...]),  # float32, 0/1
        "model.layers.0.mlp.up_proj.weight":   tensor([...]),
        "model.layers.0.mlp.down_proj.weight": tensor([...]),
        # ... one entry per masked parameter
    },
    "metadata": {  # optional but recommended
        "method": "cav",
        "sparsity_percent": 90.0,
        "model_name": "google/gemma-3-270m-it",
        # ... additional fields vary by method
    }
}
```

Per-layer keys in JSON reports use the same parameter names as the mask dict,
so reports can be cross-referenced directly.

---

## Pooling

All mask-finding algorithms default to **global pooling** (one threshold
across all layers). Pass `--local-pool` to `inference_mask_finder.py` or
`local_pool=True` in Python to switch to per-layer selection. See
`inference_mask_finder.py --help` for details.

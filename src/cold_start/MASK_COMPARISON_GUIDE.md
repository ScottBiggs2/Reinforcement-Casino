# Mask Comparison & Charting Suite

Tools for measuring how similar two sparse masks are, using two complementary metrics:
**Jaccard similarity** (structural overlap of the masks themselves) and
**linear CKA** (functional similarity of the activations each mask produces).

---

## Tools

### `mask_to_jaccard.py` — Structural mask comparison

Measures what fraction of the selected parameters are shared between two masks.
A Jaccard score of 1.0 means the masks are identical; 0.0 means they share no parameters.

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Reports both a global aggregate score (treating the entire model as one set) and
per-layer scores, which reveal whether agreement is concentrated in specific layers.

**Usage**

```bash
python src/cold_start/mask_to_jaccard.py  MASK_A.pt  MASK_B.pt  [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `mask_a` | — | First mask file (.pt) |
| `mask_b` | — | Second mask file (.pt) |
| `--output / -o` | auto | Output JSON path. Auto-generated as `jaccard_<a>_vs_<b>.json` next to the inputs if omitted. |
| `--device` | `cpu` | `cpu` or `cuda` |

**Output JSON fields**

```
jaccard.aggregate     — global Jaccard over all parameters
jaccard.mean          — mean of per-layer Jaccard scores
jaccard.min / .max    — range of per-layer scores
jaccard.n_layers      — number of layers compared
jaccard.total_intersection / .total_union
per_layer_jaccard     — {layer_name: score} dict
metadata_a / _b       — mask metadata if present in the .pt files
```

**Example**

```bash
python src/cold_start/mask_to_jaccard.py \
    mask/cold_fisher_gemma_sparsity95.pt \
    mask/cold_cav_gemma_sparsity95.pt \
    --output reports/fisher_vs_cav.json
```

---

### `mask_to_cka.py` — Functional mask comparison

Measures how similar the *activations* of two masked subnetworks are on a calibration
dataset. Unlike Jaccard, CKA captures functional equivalence — two masks that select
different parameters can still produce similar representations.

Uses **linear CKA** (Centered Kernel Alignment), which is invariant to orthogonal
transformations and isotropic scaling.

**Usage**

```bash
python src/cold_start/mask_to_cka.py  MASK_A.pt  MASK_B.pt  [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `mask_a` | — | First mask file (.pt) |
| `mask_b` | — | Second mask file (.pt) |
| `--model_name` | `google/gemma-3-270m-it` | HuggingFace model to load |
| `--compare` | `mask_vs_mask` | Comparison mode (see below) |
| `--n_samples` | `64` | Number of calibration samples |
| `--max_length` | `512` | Token length per sample |
| `--batch_size` | `4` | Forward-pass batch size |
| `--seed` | `42` | RNG seed |
| `--output / -o` | auto | Output JSON path |
| `--device` | `cpu` | `cpu` or `cuda` |

**Comparison modes (`--compare`)**

| Mode | What is compared |
|------|-----------------|
| `mask_vs_mask` | Subnetwork A activations vs subnetwork B activations (default) |
| `original_vs_a` | Full model vs subnetwork A |
| `original_vs_b` | Full model vs subnetwork B |
| `chosen_vs_rejected` | Subnetwork A on chosen texts vs rejected texts |

**Output JSON fields**

```
cka.mean / .min / .max  — summary CKA scores across layers
cka.n_layers            — number of valid layers compared
cka.n_skipped           — layers skipped due to shape mismatch
per_layer_cka           — {layer_name: score} dict (null if skipped)
compare                 — the mode used
label_a / label_b       — human-readable labels for what was compared
```

**Example**

```bash
python src/cold_start/mask_to_cka.py \
    mask/cold_fisher_gemma_sparsity95.pt \
    mask/warm_magnitude_gemma_sparsity95.pt \
    --model_name google/gemma-3-270m-it \
    --compare mask_vs_mask \
    --n_samples 128 \
    --device cuda
```

---

## When to use which tool

| Question | Tool |
|----------|------|
| Do two masks select the same parameters? | Jaccard |
| Do two masks produce similar representations? | CKA |
| How does a mask change model behavior vs the original? | CKA (`original_vs_a`) |
| Does a mask treat chosen and rejected responses differently? | CKA (`chosen_vs_rejected`) |

A high Jaccard score implies high CKA, but not vice versa — CKA can reveal functional
agreement even between structurally different masks.

---

## Output format

Both tools write a JSON report. The per-layer dicts use the same layer name keys as the
`.pt` mask files (e.g. `model.layers.0.mlp.gate_proj.weight`), so reports can be
cross-referenced directly.

To compare multiple masks at once, run pairwise comparisons and collect the
`aggregate` / `mean` scores into a matrix.

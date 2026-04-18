#!/usr/bin/env python3
"""
Probing classifier analysis for sparse mask comparison.

For each mask configuration, extracts per-layer MLP activations then trains
a linear probe per layer on several linguistic/cognitive properties.
Produces a heatmap showing which mask retains which type of knowledge at
each layer — analogous to the probing literature (e.g. Tenney et al. 2019).

Usage:
    python src/analysis/probe_analysis.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --mask_a masks/cold_snip_llama_90pct.pt \
        --mask_b masks/warm_momentum_llama_90pct.pt \
        --mask_a_label "SNIP" \
        --mask_b_label "Momentum" \
        --output_dir probe_results/

Optional flags:
    --include_baseline      also evaluate the unmasked model
    --layer_stride N        sample every N-th layer (default: 4)
    --batch_size N          inference batch size (default: 8)
    --max_length N          token length cap (default: 128)
"""

import argparse
import json
import os
import random
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from cold_start.utils.activation_hooks import FeatureExtractor


# ---------------------------------------------------------------------------
# HuggingFace probe dataset loading
# ---------------------------------------------------------------------------
# Each property uses a standard HF dataset, balanced to N examples per class.
# Default: 200 per class (400 per property, 1600 total) — much larger than
# the previous 30-per-class hand-written set.
# ---------------------------------------------------------------------------

def _load_hf_probe_datasets(samples_per_class: int = None,
                            cache_dir: str = None, seed: int = 42) -> dict:
    """Load probe datasets from HuggingFace and return in the same format
    as the old hardcoded PROBE_DATASETS dict.

    Args:
        samples_per_class: Max samples per class. None = use all available data
                           (balanced to the smallest class across all properties).
        cache_dir: HuggingFace datasets cache directory.
        seed: Random seed for shuffling order (not for subsetting when using
              full data).

    Returns:
        {property_name: {"description": str, "examples": [(text, label), ...]}}
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    datasets = {}

    # ── 1. Syntax: BLiMP subject-verb agreement ──────────────────────────
    print(f"[data] Loading syntax probe (BLiMP regular_plural_subject_verb_agreement_1)...")
    try:
        blimp = load_dataset(
            "nyu-mll/blimp", "regular_plural_subject_verb_agreement_1",
            split="train", cache_dir=cache_dir, trust_remote_code=True,
        )
        good = [(row["sentence_good"], 1) for row in blimp]
        bad = [(row["sentence_bad"], 0) for row in blimp]
        datasets["syntax"] = {
            "description": "Subject-verb agreement grammaticality (BLiMP; 0=bad, 1=good)",
            "pos": good, "neg": bad,
        }
        print(f"[data]   syntax: {len(good)} good, {len(bad)} bad")
    except Exception as e:
        print(f"[data]   WARNING: BLiMP load failed ({e}), falling back to CoLA")
        cola = load_dataset("glue", "cola", split="validation", cache_dir=cache_dir)
        pos = [(row["sentence"], 1) for row in cola if row["label"] == 1]
        neg = [(row["sentence"], 0) for row in cola if row["label"] == 0]
        datasets["syntax"] = {
            "description": "Linguistic acceptability (CoLA; 0=unacceptable, 1=acceptable)",
            "pos": pos, "neg": neg,
        }
        print(f"[data]   syntax (CoLA fallback): {len(pos)} pos, {len(neg)} neg")

    # ── 2. Semantics: SST-2 sentiment ────────────────────────────────────
    print(f"[data] Loading semantics probe (SST-2)...")
    sst2 = load_dataset("glue", "sst2", split="validation", cache_dir=cache_dir)
    pos = [(row["sentence"], 1) for row in sst2 if row["label"] == 1]
    neg = [(row["sentence"], 0) for row in sst2 if row["label"] == 0]
    datasets["semantics"] = {
        "description": "Sentiment polarity (SST-2; 0=negative, 1=positive)",
        "pos": pos, "neg": neg,
    }
    print(f"[data]   semantics: {len(pos)} pos, {len(neg)} neg")

    # ── 3. Factual: AG News (World=0 vs Sci/Tech=1) ─────────────────────
    print(f"[data] Loading factual probe (AG News)...")
    agnews = load_dataset("fancyzhx/ag_news", split="test", cache_dir=cache_dir)
    world = [(row["text"], 0) for row in agnews if row["label"] == 0]
    scitech = [(row["text"], 1) for row in agnews if row["label"] == 3]
    datasets["factual"] = {
        "description": "Topic classification (AG News; 0=World, 1=Sci/Tech)",
        "pos": scitech, "neg": world,
    }
    print(f"[data]   factual: {len(world)} world, {len(scitech)} sci/tech")

    # ── 4. Math: GSM8K-style arithmetic (correct vs incorrect) ───────────
    print(f"[data] Loading math probe (GSM8K)...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
        questions = [row["question"] for row in gsm8k]
        rng.shuffle(questions)

        # Split on sentence terminators while keeping them on the preceding
        # sentence (avoids losing "?" from the final question).
        def _split_sentences(q: str) -> list:
            parts = re.split(r"(?<=[.!?])\s+", q.strip())
            return [p for p in parts if p]

        def _shuffled_distinct(sents: list, max_tries: int = 10):
            for _ in range(max_tries):
                cand = sents[:]
                rng.shuffle(cand)
                if cand != sents:
                    return cand
            return None

        # Only keep questions with enough structure to meaningfully shuffle.
        shuffleable = [(q, _split_sentences(q)) for q in questions]
        shuffleable = [(q, s) for q, s in shuffleable if len(s) >= 3]

        half = len(shuffleable) // 2
        real_qs = [(q, 1) for q, _ in shuffleable[:half]]
        fake_qs = []
        for _, sents in shuffleable[half:half * 2]:
            shuffled = _shuffled_distinct(sents)
            if shuffled is not None:
                fake_qs.append((" ".join(shuffled), 0))

        datasets["math"] = {
            "description": "Math coherence (GSM8K; 0=shuffled, 1=coherent)",
            "pos": real_qs, "neg": fake_qs,
        }
        print(f"[data]   math: {len(real_qs)} real, {len(fake_qs)} shuffled "
              f"(filtered from {len(questions)} to questions with ≥3 sentences)")
    except Exception as e:
        print(f"[data]   WARNING: GSM8K load failed ({e}), using MMLU abstract_algebra")
        mmlu = load_dataset("cais/mmlu", "abstract_algebra", split="test", cache_dir=cache_dir)
        classA = [(row["question"], 0) for row in mmlu if row["answer"] == 0]
        classB = [(row["question"], 1) for row in mmlu if row["answer"] != 0]
        datasets["math"] = {
            "description": "Math reasoning (MMLU abstract_algebra; binary split)",
            "pos": classB, "neg": classA,
        }
        print(f"[data]   math (MMLU fallback): {len(classA)} classA, {len(classB)} classB")

    # ── Balance: use full data, equalized to smallest class ──────────────
    # Find the global min class size across all properties
    global_min = min(
        min(len(d["pos"]), len(d["neg"])) for d in datasets.values()
    )
    if samples_per_class is not None:
        global_min = min(global_min, samples_per_class)

    print(f"[data] Balancing all properties to {global_min} per class "
          f"({global_min * 2} per property, {global_min * 2 * len(datasets)} total)")

    result = {}
    for prop_name, d in datasets.items():
        pos = d["pos"][:global_min]
        neg = d["neg"][:global_min]
        examples = pos + neg
        rng.shuffle(examples)
        result[prop_name] = {
            "description": d["description"],
            "examples": examples,
        }

    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _layer_index(layer_name: str) -> int:
    """Extract integer layer index from names like 'model.layers.12.mlp.down_proj'."""
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


def load_mask(path: str) -> dict:
    """Load a mask file, handling both raw dicts and wrapped {masks: ...} format."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"]
    return data


@contextmanager
def apply_mask(model, mask_dict: dict):
    """Context manager: temporarily zero out weights according to mask_dict.

    mask_dict maps param_name -> binary tensor (1=keep, 0=prune).
    Original weights are restored on exit regardless of exceptions.
    """
    originals = {}
    try:
        for name, param in model.named_parameters():
            if name in mask_dict:
                m = mask_dict[name]
                # Strict {0,1} check: non-binary masks would silently scale
                # weights, corrupting the baseline restore below.
                if not torch.all((m == 0) | (m == 1)):
                    raise ValueError(
                        f"Mask for {name} contains non-binary values; "
                        f"refusing to apply to avoid precision loss."
                    )
                originals[name] = param.data.clone()
                param.data.mul_(m.to(param.device, dtype=param.dtype))
        yield
    finally:
        for name, param in model.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])


@contextmanager
def no_mask():
    """Dummy context manager for the unmasked baseline."""
    yield


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------

def train_probes(activations_by_layer: dict, labels: np.ndarray, cv: int = 5) -> dict:
    """Train a linear probe per layer with full diagnostics.

    Returns {layer_name: {"test", "std", "train", "gap", "converged", "degenerate"}}.
    Scaler is inside a Pipeline so per-fold leakage is avoided. Degenerate features
    (all-zero / NaN activations) are detected and skipped with NaN results instead
    of raising from inside LR.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for layer_name, acts in activations_by_layer.items():
        X = acts.float().numpy()

        nan_frac = float(np.isnan(X).mean())
        x_std = float(X.std())
        degenerate = nan_frac > 0 or x_std < 1e-8

        if degenerate:
            results[layer_name] = {
                "test": float("nan"),
                "std": float("nan"),
                "train": float("nan"),
                "gap": float("nan"),
                "converged": False,
                "degenerate": True,
            }
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=1.0,
                max_iter=2000,
                random_state=42,
            )),
        ])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            # n_jobs=1: cross_validate parallelism with Pipeline on 14336-d
            # features copies X per worker, which blew memory past 200GB on
            # a 32-core node. Serial folds are fast enough here.
            out = cross_validate(
                pipe, X, labels, cv=skf,
                scoring="accuracy",
                return_train_score=True,
                n_jobs=1,
            )
            converged = not any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )

        test_scores = out["test_score"]
        train_scores = out["train_score"]
        results[layer_name] = {
            "test": float(test_scores.mean()),
            "std": float(test_scores.std()),
            "train": float(train_scores.mean()),
            "gap": float(train_scores.mean() - test_scores.mean()),
            "converged": converged,
            "degenerate": False,
        }

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_probe_heatmap(
    results_by_config: dict,
    sample_indices: list,
    property_order: list,
    output_path: str,
):
    """Plot a grid of heatmaps: one panel per mask config.

    Args:
        results_by_config: {config_label: {prop_name: {layer_name: accuracy}}}
        sample_indices: list of integer layer indices that were sampled
        property_order: list of property names in display order
        output_path: path for the output PNG
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    n_configs = len(results_by_config)
    n_props = len(property_order)
    n_layers = len(sample_indices)

    layer_labels = [f"layer {i}" for i in sample_indices]
    # Colormap: below chance → red, chance (0.5) → yellow, 1.0 → dark green.
    # vmin must go below 0.5 so below-chance results are visibly distinct from
    # chance rather than clipped to the same colour.
    cmap = plt.cm.RdYlGn
    vmin, vmax = 0.3, 1.0
    sample_idx_to_col = {idx: i for i, idx in enumerate(sample_indices)}

    fig_width = max(10, 4 * n_configs + 2)
    fig, axes = plt.subplots(
        1, n_configs,
        figsize=(fig_width, n_props * 0.9 + 2.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (config_label, prop_results) in zip(axes, results_by_config.items()):
        # Build [n_props, n_layers] accuracy matrix. Columns are positioned
        # by layer index, not by iteration order — the two agree today but
        # decoupling them prevents silent misalignment if upstream filtering
        # ever drops a layer.
        mat = np.full((n_props, n_layers), np.nan)
        for pi, prop in enumerate(property_order):
            if prop not in prop_results:
                continue
            for lname, acc in prop_results[prop].items():
                idx = _layer_index(lname)
                col = sample_idx_to_col.get(idx)
                if col is not None:
                    mat[pi, col] = acc["test"] if isinstance(acc, dict) else acc

        # NaN cells render as chance (neutral yellow). Real below-chance
        # accuracies are left unclipped down to vmin so they show as red.
        display_mat = np.where(np.isnan(mat), 0.5, np.clip(mat, vmin, vmax))

        im = ax.imshow(display_mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_props))
        ax.set_yticklabels(property_order, fontsize=11)
        ax.set_title(config_label, fontsize=13, fontweight="bold", pad=10)

        # Annotate cells
        for pi in range(n_props):
            for li in range(n_layers):
                val = mat[pi, li]
                if np.isnan(val):
                    continue
                txt_color = "white" if val < 0.6 or val > 0.88 else "black"
                ax.text(
                    li, pi, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold",
                )

    # Shared colorbar
    fig.subplots_adjust(right=0.86, wspace=0.35)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("probe accuracy", fontsize=10, labelpad=8)
    cbar.set_ticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Mark chance explicitly so readers can spot below-chance cells.
    cbar.ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    cbar_ax.text(
        0.5, -0.04, "low\n(knowledge lost)",
        ha="center", va="top", transform=cbar_ax.transAxes, fontsize=8,
    )
    cbar_ax.text(
        0.5, 1.04, "high\n(knowledge retained)",
        ha="center", va="bottom", transform=cbar_ax.transAxes, fontsize=8,
    )

    fig.suptitle("Probing Classifier Analysis: Knowledge Retention per Layer", fontsize=14, y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved heatmap → {output_path}")
    plt.close()


def plot_delta_heatmap(
    results_by_config: dict,
    baseline_key: str,
    sample_indices: list,
    property_order: list,
    output_path: str,
):
    """Plot Δaccuracy (masked − baseline) heatmap for each non-baseline config.

    Args:
        results_by_config: {config_label: {prop_name: {layer_name: accuracy}}}
        baseline_key: key in results_by_config for the unmasked baseline
        sample_indices: list of integer layer indices that were sampled
        property_order: list of property names in display order
        output_path: path for the output PNG
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    baseline_results = results_by_config[baseline_key]
    mask_configs = {k: v for k, v in results_by_config.items() if k != baseline_key}

    if not mask_configs:
        print("[plot] No mask configs to compare against baseline; skipping delta plot.")
        return

    n_configs = len(mask_configs)
    n_props = len(property_order)
    n_layers = len(sample_indices)
    layer_labels = [f"layer {i}" for i in sample_indices]

    sample_idx_to_col = {idx: i for i, idx in enumerate(sample_indices)}

    def _build_matrix(prop_results):
        mat = np.full((n_props, n_layers), np.nan)
        for pi, prop in enumerate(property_order):
            if prop not in prop_results:
                continue
            for lname, acc in prop_results[prop].items():
                idx = _layer_index(lname)
                col = sample_idx_to_col.get(idx)
                if col is not None:
                    mat[pi, col] = acc["test"] if isinstance(acc, dict) else acc
        return mat

    baseline_mat = _build_matrix(baseline_results)

    # Compute all deltas to find symmetric color range
    delta_mats = {}
    for config_label, prop_results in mask_configs.items():
        delta_mats[config_label] = _build_matrix(prop_results) - baseline_mat

    all_deltas = np.concatenate([m[~np.isnan(m)] for m in delta_mats.values()])
    abs_max = max(0.05, np.nanmax(np.abs(all_deltas)))  # at least ±0.05 range

    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig_width = max(10, 4 * n_configs + 2)
    fig, axes = plt.subplots(
        1, n_configs,
        figsize=(fig_width, n_props * 0.9 + 2.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (config_label, delta_mat) in zip(axes, delta_mats.items()):
        display_mat = np.where(np.isnan(delta_mat), 0.0, delta_mat)
        im = ax.imshow(display_mat, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_props))
        ax.set_yticklabels(property_order, fontsize=11)
        ax.set_title(config_label, fontsize=13, fontweight="bold", pad=10)

        for pi in range(n_props):
            for li in range(n_layers):
                val = delta_mat[pi, li]
                if np.isnan(val):
                    continue
                txt_color = "white" if abs(val) > abs_max * 0.65 else "black"
                sign = "+" if val > 0 else ""
                ax.text(
                    li, pi, f"{sign}{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold",
                )

    fig.subplots_adjust(right=0.86, wspace=0.35)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Δ accuracy (masked − baseline)", fontsize=10, labelpad=8)
    cbar_ax.text(
        0.5, -0.04, "knowledge\nlost",
        ha="center", va="top", transform=cbar_ax.transAxes, fontsize=8,
    )
    cbar_ax.text(
        0.5, 1.04, "knowledge\ngained",
        ha="center", va="bottom", transform=cbar_ax.transAxes, fontsize=8,
    )

    fig.suptitle("Δ Probe Accuracy: Mask vs Baseline", fontsize=14, y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved delta heatmap → {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Probing classifier analysis for mask comparison")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HuggingFace model name or local path")
    p.add_argument("--mask_a", required=True, help="Path to mask A .pt file")
    p.add_argument("--mask_b", required=True, help="Path to mask B .pt file")
    p.add_argument("--mask_a_label", default="Mask A", help="Display label for mask A")
    p.add_argument("--mask_b_label", default="Mask B", help="Display label for mask B")
    p.add_argument("--include_baseline", action="store_true", default=True,
                   help="Run unmasked model as baseline (default: on, needed for delta plots)")
    p.add_argument("--no_baseline", action="store_true",
                   help="Skip the unmasked baseline (disables delta heatmap)")
    p.add_argument("--output_dir", default="probe_results",
                   help="Directory for JSON and PNG outputs")
    p.add_argument("--layer_stride", type=int, default=4,
                   help="Plot every N-th layer (default: 4). Use 1 for all layers.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256,
                   help="Token length cap. 256 covers ~99%% of AG News and "
                        "most GSM8K questions; lower values systematically "
                        "penalise factual/math probes.")
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Cross-validation folds for probe accuracy (default: 5)")
    p.add_argument("--samples_per_class", type=int, default=None,
                   help="Samples per class per property (default: None = use all available data)")
    p.add_argument("--dataset_cache_dir", default=None,
                   help="HuggingFace datasets cache directory")
    p.add_argument("--probe_cache", default=None,
                   help="Path to cached probe dataset JSON. If set, overrides auto-detection. "
                        "Ensures all runs use identical probe samples.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_baseline:
        args.include_baseline = False
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load probe datasets (with disk caching for cross-run consistency)
    # ------------------------------------------------------------------
    cache_dir = args.dataset_cache_dir or os.environ.get("HF_DATASETS_CACHE")
    # Cache version tag — bump when dataset construction logic changes so old
    # caches are invalidated instead of silently reused.
    _CACHE_VERSION = "v2"
    if args.probe_cache:
        probe_cache_path = args.probe_cache
    else:
        n_tag = args.samples_per_class if args.samples_per_class is not None else "all"
        cache_name = f"probe_dataset_cache_{_CACHE_VERSION}_n{n_tag}_seed42.json"
        probe_cache_path = os.path.join(args.output_dir, os.pardir, cache_name)
        probe_cache_path = os.path.normpath(probe_cache_path)

    if os.path.exists(probe_cache_path):
        print(f"[data] Loading cached probe dataset from {probe_cache_path}")
        with open(probe_cache_path) as f:
            PROBE_DATASETS = json.load(f)
        # Convert lists back to tuples
        for prop in PROBE_DATASETS:
            PROBE_DATASETS[prop]["examples"] = [
                (t, l) for t, l in PROBE_DATASETS[prop]["examples"]
            ]
        print(f"[data] Loaded {len(PROBE_DATASETS)} properties from cache")
    else:
        PROBE_DATASETS = _load_hf_probe_datasets(
            samples_per_class=args.samples_per_class,
            cache_dir=cache_dir,
        )
        # Cache to disk for subsequent runs
        os.makedirs(os.path.dirname(probe_cache_path), exist_ok=True)
        with open(probe_cache_path, "w") as f:
            json.dump(PROBE_DATASETS, f)
        print(f"[data] Cached probe dataset → {probe_cache_path}")

    expected_n = None
    for prop_name, prop_data in PROBE_DATASETS.items():
        n = len(prop_data["examples"])
        if expected_n is None:
            expected_n = n
        assert n == expected_n, (
            f"Property '{prop_name}' has {n} examples; expected {expected_n}"
        )
        labels = [lbl for _, lbl in prop_data["examples"]]
        n_pos = sum(1 for l in labels if l == 1)
        n_neg = sum(1 for l in labels if l == 0)
        assert n_pos == n_neg, (
            f"Property '{prop_name}' is imbalanced: {n_pos} pos vs {n_neg} neg"
        )

    # ------------------------------------------------------------------
    # Concatenate all texts for a single inference pass
    # ------------------------------------------------------------------
    all_texts = []
    property_slices = {}
    for prop_name, prop_data in PROBE_DATASETS.items():
        texts = [t for t, _ in prop_data["examples"]]
        property_slices[prop_name] = slice(len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)

    print(f"[main] {len(PROBE_DATASETS)} probe properties, "
          f"{expected_n} examples each, {len(all_texts)} texts total")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n[main] Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Count MLP layers and build sample index list
    n_mlp_layers = sum(
        1 for name, _ in model.named_modules() if name.endswith("down_proj")
    )
    sample_indices = list(range(0, n_mlp_layers, args.layer_stride))
    if (n_mlp_layers - 1) not in sample_indices:
        sample_indices.append(n_mlp_layers - 1)
    sample_index_set = set(sample_indices)
    print(f"[main] Model has {n_mlp_layers} MLP layers; "
          f"sampling {len(sample_indices)} layers: {sample_indices}")

    # ------------------------------------------------------------------
    # Load masks
    # ------------------------------------------------------------------
    print(f"\n[main] Loading mask A: {args.mask_a}")
    mask_a = load_mask(args.mask_a)
    print(f"[main] Loading mask B: {args.mask_b}")
    mask_b = load_mask(args.mask_b)

    # ------------------------------------------------------------------
    # Build configs
    # ------------------------------------------------------------------
    configs = {}
    if args.include_baseline:
        configs["Baseline\n(no mask)"] = None
    configs[args.mask_a_label] = mask_a
    configs[args.mask_b_label] = mask_b

    # ------------------------------------------------------------------
    # Activation collection + probe training
    # ------------------------------------------------------------------
    extractor = FeatureExtractor().register(model)
    results_by_config = {}

    for config_label, mask_dict in configs.items():
        label_clean = config_label.replace("\n", " ")
        print(f"\n[main] ===== {label_clean} =====")

        ctx = apply_mask(model, mask_dict) if mask_dict is not None else no_mask()

        with ctx:
            device = next(model.parameters()).device
            activations = extractor.collect(
                model, tokenizer, all_texts, device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

        # Filter to sampled layers only
        sampled_acts = {
            name: acts
            for name, acts in activations.items()
            if _layer_index(name) in sample_index_set
        }

        prop_results = {}
        for prop_name, prop_data in PROBE_DATASETS.items():
            slc = property_slices[prop_name]
            labels_arr = np.array([lbl for _, lbl in prop_data["examples"]])

            # Slice activations for this property's texts
            prop_acts = {name: acts[slc] for name, acts in sampled_acts.items()}

            layer_accs = train_probes(prop_acts, labels_arr, cv=args.cv_folds)
            prop_results[prop_name] = layer_accs

            test_vals = [v["test"] for v in layer_accs.values()]
            train_vals = [v["train"] for v in layer_accs.values()]
            n_notconv = sum(1 for v in layer_accs.values() if not v["converged"])
            n_degen = sum(1 for v in layer_accs.values() if v["degenerate"])
            print(
                f"  {prop_name:12s}: "
                f"test={np.nanmean(test_vals):.3f}  "
                f"train={np.nanmean(train_vals):.3f}  "
                f"gap={np.nanmean(train_vals) - np.nanmean(test_vals):+.3f}  "
                f"[min={np.nanmin(test_vals):.3f} max={np.nanmax(test_vals):.3f}]  "
                f"non_conv={n_notconv}/{len(layer_accs)}  "
                f"degen={n_degen}/{len(layer_accs)}"
            )

        results_by_config[config_label] = prop_results

        # Config-level health summary — bubbles up issues across all props/layers
        total_probes = sum(len(v) for v in prop_results.values())
        total_notconv = sum(
            1 for prop in prop_results.values()
            for v in prop.values() if not v["converged"]
        )
        total_degen = sum(
            1 for prop in prop_results.values()
            for v in prop.values() if v["degenerate"]
        )
        if total_notconv or total_degen:
            print(
                f"  ⚠️  [{label_clean}] {total_notconv}/{total_probes} did not converge, "
                f"{total_degen}/{total_probes} degenerate"
            )

    extractor.remove()

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(results_by_config, f, indent=2)
    print(f"\n[main] Saved results JSON → {json_path}")

    # ------------------------------------------------------------------
    # Plot heatmaps
    # ------------------------------------------------------------------
    plot_path = os.path.join(args.output_dir, "probe_heatmap.png")
    plot_probe_heatmap(
        results_by_config,
        sample_indices=sample_indices,
        property_order=list(PROBE_DATASETS.keys()),
        output_path=plot_path,
    )

    baseline_key = "Baseline\n(no mask)"
    if baseline_key in results_by_config:
        delta_path = os.path.join(args.output_dir, "probe_delta_heatmap.png")
        plot_delta_heatmap(
            results_by_config,
            baseline_key=baseline_key,
            sample_indices=sample_indices,
            property_order=list(PROBE_DATASETS.keys()),
            output_path=delta_path,
        )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()

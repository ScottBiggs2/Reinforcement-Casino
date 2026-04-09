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
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

DEFAULT_SAMPLES_PER_CLASS = 200

def _load_hf_probe_datasets(samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
                            cache_dir: str = None, seed: int = 42) -> dict:
    """Load probe datasets from HuggingFace and return in the same format
    as the old hardcoded PROBE_DATASETS dict.

    Returns:
        {property_name: {"description": str, "examples": [(text, label), ...]}}
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    n = samples_per_class

    datasets = {}

    # ── 1. Syntax: BLiMP subject-verb agreement ──────────────────────────
    # BLiMP provides minimal pairs; each row has a good and bad sentence.
    print(f"[data] Loading syntax probe (BLiMP subject_verb_number_local)...")
    try:
        blimp = load_dataset(
            "nyu-mll/blimp", "subject_verb_number_local",
            split="train", cache_dir=cache_dir, trust_remote_code=True,
        )
        # good sentence → 1 (grammatical), bad sentence → 0 (ungrammatical)
        good = [(row["sentence_good"], 1) for row in blimp]
        bad = [(row["sentence_bad"], 0) for row in blimp]
        rng.shuffle(good)
        rng.shuffle(bad)
        examples = good[:n] + bad[:n]
        rng.shuffle(examples)
        datasets["syntax"] = {
            "description": "Subject-verb agreement grammaticality (BLiMP; 0=bad, 1=good)",
            "examples": examples,
        }
        print(f"[data]   syntax: {len(examples)} examples ({n} per class)")
    except Exception as e:
        print(f"[data]   WARNING: BLiMP load failed ({e}), falling back to CoLA")
        cola = load_dataset("glue", "cola", split="validation", cache_dir=cache_dir)
        pos = [(row["sentence"], 1) for row in cola if row["label"] == 1]
        neg = [(row["sentence"], 0) for row in cola if row["label"] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        k = min(n, len(pos), len(neg))
        examples = pos[:k] + neg[:k]
        rng.shuffle(examples)
        datasets["syntax"] = {
            "description": "Linguistic acceptability (CoLA; 0=unacceptable, 1=acceptable)",
            "examples": examples,
        }
        print(f"[data]   syntax (CoLA fallback): {len(examples)} examples ({k} per class)")

    # ── 2. Semantics: SST-2 sentiment ────────────────────────────────────
    print(f"[data] Loading semantics probe (SST-2)...")
    sst2 = load_dataset("glue", "sst2", split="validation", cache_dir=cache_dir)
    pos = [(row["sentence"], 1) for row in sst2 if row["label"] == 1]
    neg = [(row["sentence"], 0) for row in sst2 if row["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n, len(pos), len(neg))
    examples = pos[:k] + neg[:k]
    rng.shuffle(examples)
    datasets["semantics"] = {
        "description": "Sentiment polarity (SST-2; 0=negative, 1=positive)",
        "examples": examples,
    }
    print(f"[data]   semantics: {len(examples)} examples ({k} per class)")

    # ── 3. Factual: AG News (World=0 vs Sci/Tech=1) ─────────────────────
    print(f"[data] Loading factual probe (AG News)...")
    agnews = load_dataset("fancyzhx/ag_news", split="test", cache_dir=cache_dir)
    # AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
    world = [(row["text"], 0) for row in agnews if row["label"] == 0]
    scitech = [(row["text"], 1) for row in agnews if row["label"] == 3]
    rng.shuffle(world)
    rng.shuffle(scitech)
    k = min(n, len(world), len(scitech))
    examples = world[:k] + scitech[:k]
    rng.shuffle(examples)
    datasets["factual"] = {
        "description": "Topic classification (AG News; 0=World, 1=Sci/Tech)",
        "examples": examples,
    }
    print(f"[data]   factual: {len(examples)} examples ({k} per class)")

    # ── 4. Math: GSM8K-style arithmetic (correct vs incorrect) ───────────
    print(f"[data] Loading math probe (GSM8K)...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
        # Use the question as text; label 1 = real question, 0 = shuffled question
        questions = [row["question"] for row in gsm8k]
        rng.shuffle(questions)
        real_qs = [(q, 1) for q in questions[:n]]
        # Create "wrong" examples by shuffling sentences within questions
        fake_qs = []
        for q in questions[n:n*2]:
            sentences = q.split(". ")
            rng.shuffle(sentences)
            fake_qs.append((". ".join(sentences), 0))
        k = min(n, len(real_qs), len(fake_qs))
        examples = real_qs[:k] + fake_qs[:k]
        rng.shuffle(examples)
        datasets["math"] = {
            "description": "Math coherence (GSM8K; 0=shuffled, 1=coherent)",
            "examples": examples,
        }
        print(f"[data]   math: {len(examples)} examples ({k} per class)")
    except Exception as e:
        print(f"[data]   WARNING: GSM8K load failed ({e}), using MMLU abstract_algebra")
        mmlu = load_dataset("cais/mmlu", "abstract_algebra", split="test", cache_dir=cache_dir)
        # Use question text; label = whether answer is A (0) or not (1) — simple binary split
        classA = [(row["question"], 0) for row in mmlu if row["answer"] == 0]
        classB = [(row["question"], 1) for row in mmlu if row["answer"] != 0]
        rng.shuffle(classA)
        rng.shuffle(classB)
        k = min(n, len(classA), len(classB))
        examples = classA[:k] + classB[:k]
        rng.shuffle(examples)
        datasets["math"] = {
            "description": "Math reasoning (MMLU abstract_algebra; binary split)",
            "examples": examples,
        }
        print(f"[data]   math (MMLU fallback): {len(examples)} examples ({k} per class)")

    # ── Equalize sizes across properties ─────────────────────────────────
    min_total = min(len(d["examples"]) for d in datasets.values())
    # Make sure it's even (balanced)
    min_per_class = min_total // 2
    for prop_name in datasets:
        exs = datasets[prop_name]["examples"]
        class0 = [e for e in exs if e[1] == 0][:min_per_class]
        class1 = [e for e in exs if e[1] == 1][:min_per_class]
        balanced = class0 + class1
        rng.shuffle(balanced)
        datasets[prop_name]["examples"] = balanced

    final_n = len(next(iter(datasets.values()))["examples"])
    print(f"[data] All properties equalized to {final_n} examples "
          f"({final_n // 2} per class)")

    return datasets


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
                originals[name] = param.data.clone()
                param.data.mul_(mask_dict[name].to(param.device, dtype=param.dtype))
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
    """Train a linear probe per layer and return cross-validated accuracy.

    Args:
        activations_by_layer: {layer_name: Tensor[N, D]}
        labels: np.array of shape [N] with binary labels
        cv: number of cross-validation folds

    Returns:
        {layer_name: float mean_cv_accuracy}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for layer_name, acts in activations_by_layer.items():
        X = acts.float().numpy()
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=1.0,
            max_iter=500,
            random_state=42,
        )
        scores = cross_val_score(clf, X_sc, labels, cv=skf, scoring="accuracy")
        results[layer_name] = float(scores.mean())

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
    # Colormap: chance (0.5) → yellow, 1.0 → dark green; below chance → red
    cmap = plt.cm.RdYlGn
    vmin, vmax = 0.5, 1.0

    fig_width = max(10, 4 * n_configs + 2)
    fig, axes = plt.subplots(
        1, n_configs,
        figsize=(fig_width, n_props * 0.9 + 2.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (config_label, prop_results) in zip(axes, results_by_config.items()):
        # Build [n_props, n_layers] accuracy matrix
        mat = np.full((n_props, n_layers), np.nan)
        for pi, prop in enumerate(property_order):
            if prop not in prop_results:
                continue
            layer_map = prop_results[prop]
            sorted_layer_names = sorted(layer_map.keys(), key=_layer_index)
            for li, lname in enumerate(sorted_layer_names):
                if li < n_layers:
                    mat[pi, li] = layer_map[lname]

        # Clamp values below chance to vmin for display purposes
        display_mat = np.where(np.isnan(mat), vmin, np.clip(mat, vmin, vmax))

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
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
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

    def _build_matrix(prop_results):
        mat = np.full((n_props, n_layers), np.nan)
        for pi, prop in enumerate(property_order):
            if prop not in prop_results:
                continue
            sorted_names = sorted(prop_results[prop].keys(), key=_layer_index)
            for li, lname in enumerate(sorted_names):
                if li < n_layers:
                    mat[pi, li] = prop_results[prop][lname]
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
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Cross-validation folds for probe accuracy (default: 5)")
    p.add_argument("--samples_per_class", type=int, default=DEFAULT_SAMPLES_PER_CLASS,
                   help=f"Samples per class per property from HF datasets (default: {DEFAULT_SAMPLES_PER_CLASS})")
    p.add_argument("--dataset_cache_dir", default=None,
                   help="HuggingFace datasets cache directory")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_baseline:
        args.include_baseline = False
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load probe datasets from HuggingFace
    # ------------------------------------------------------------------
    cache_dir = args.dataset_cache_dir or os.environ.get("HF_DATASETS_CACHE")
    PROBE_DATASETS = _load_hf_probe_datasets(
        samples_per_class=args.samples_per_class,
        cache_dir=cache_dir,
    )

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

            vals = list(layer_accs.values())
            print(f"  {prop_name:12s}: mean={np.mean(vals):.3f}  "
                  f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

        results_by_config[config_label] = prop_results

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

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
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.cold_start.probe_builtin_datasets import (
    PROBE_DATASETS,
    build_concatenated_texts_and_slices,
    layer_index_from_hook_name,
    train_linear_probes_cv,
    validate_probe_datasets,
)
from src.cold_start.utils.activation_hooks import FeatureExtractor

_layer_index = layer_index_from_hook_name


def train_probes(activations_by_layer: dict, labels: np.ndarray, cv: int = 5) -> dict:
    """Backward-compatible name: delegate to shared ``train_linear_probes_cv``."""
    acc, _diag = train_linear_probes_cv(activations_by_layer, labels, cv=cv)
    return acc


def load_mask(path: str) -> dict:
    """Load a mask file, handling both raw dicts and wrapped ``{masks: ...}`` format."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"]
    return data


@contextmanager
def apply_mask(model, mask_dict: dict):
    """Temporarily multiply weights by binary mask; restore on exit."""
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
    yield


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
    p.add_argument("--include_baseline", action="store_true",
                   help="Also run on the unmasked model as a baseline panel")
    p.add_argument("--output_dir", default="probe_results",
                   help="Directory for JSON and PNG outputs")
    p.add_argument("--layer_stride", type=int, default=4,
                   help="Plot every N-th layer (default: 4). Use 1 for all layers.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Cross-validation folds for probe accuracy (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    validate_probe_datasets()
    prop_keys = list(PROBE_DATASETS.keys())
    all_texts, property_slices, labels_by_prop = build_concatenated_texts_and_slices(prop_keys)
    expected_n = len(PROBE_DATASETS[prop_keys[0]]["examples"])
    print(
        f"[main] {len(PROBE_DATASETS)} probe properties, "
        f"{expected_n} examples each, {len(all_texts)} texts total"
    )

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
        for prop_name in PROBE_DATASETS:
            slc = property_slices[prop_name]
            labels_arr = labels_by_prop[prop_name]

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
    # Plot heatmap
    # ------------------------------------------------------------------
    plot_path = os.path.join(args.output_dir, "probe_heatmap.png")
    plot_probe_heatmap(
        results_by_config,
        sample_indices=sample_indices,
        property_order=list(PROBE_DATASETS.keys()),
        output_path=plot_path,
    )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()

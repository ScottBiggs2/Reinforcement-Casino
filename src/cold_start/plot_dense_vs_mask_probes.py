#!/usr/bin/env python3
"""Multi-panel heatmaps for dense-vs-mask builtin probe reports.

Consumes ``dense_vs_mask_probes.json`` written by ``dense_vs_mask_probes.py`` and
produces two paper-ready figures:

1) Accuracy heatmaps: Baseline (dense) + one panel per mask
2) Delta heatmaps: (masked - dense) for each mask

We only target the builtin properties (syntax/semantics/factual/math) per the plan.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_LAYER_IDX = re.compile(r"model\.layers\.(\d+)\.")


def _layer_index(name: str) -> Optional[int]:
    m = _LAYER_IDX.search(str(name))
    return int(m.group(1)) if m else None


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_layers(properties: List[Dict[str, Any]]) -> List[str]:
    """Union of layer keys across properties, sorted by decoder layer index."""
    keys = set()
    for p in properties:
        per_layer = p.get("per_layer") or {}
        keys |= set(per_layer.keys())
    layers = sorted(keys, key=lambda k: (_layer_index(k) is None, _layer_index(k) or 0, str(k)))
    return layers


def _property_order(props: List[Dict[str, Any]]) -> List[str]:
    # Stable, expected ordering (only builtins matter).
    want = ["syntax", "semantics", "factual", "math"]
    seen = {p.get("property") for p in props}
    ordered = [p for p in want if p in seen]
    # Append any extras at end, stable
    ordered += [p.get("property") for p in props if p.get("property") not in ordered]
    return [x for x in ordered if isinstance(x, str)]


def _matrix_accuracy(
    props: List[Dict[str, Any]],
    prop_order: List[str],
    layers: List[str],
    *,
    panel: str,  # "baseline" or mask label
) -> np.ndarray:
    """Return [n_props, n_layers] accuracy matrix."""
    mat = np.full((len(prop_order), len(layers)), np.nan, dtype=float)
    by_prop = {p.get("property"): p for p in props}
    for pi, prop in enumerate(prop_order):
        block = by_prop.get(prop) or {}
        per_layer = block.get("per_layer") or {}
        for li, lname in enumerate(layers):
            row = per_layer.get(lname) or {}
            if panel == "baseline":
                v = row.get("dense_test_accuracy")
            else:
                v = ((row.get("masks") or {}).get(panel) or {}).get("masked_test_accuracy")
            if isinstance(v, (int, float)) and v == v:
                mat[pi, li] = float(v)
    return mat


def _matrix_delta(
    props: List[Dict[str, Any]],
    prop_order: List[str],
    layers: List[str],
    *,
    mask_label: str,
) -> np.ndarray:
    """Return [n_props, n_layers] delta matrix (masked - dense)."""
    mat = np.full((len(prop_order), len(layers)), np.nan, dtype=float)
    by_prop = {p.get("property"): p for p in props}
    for pi, prop in enumerate(prop_order):
        block = by_prop.get(prop) or {}
        per_layer = block.get("per_layer") or {}
        for li, lname in enumerate(layers):
            row = per_layer.get(lname) or {}
            v = ((row.get("masks") or {}).get(mask_label) or {}).get("delta_vs_dense")
            if isinstance(v, (int, float)) and v == v:
                mat[pi, li] = float(v)
    return mat


def _plot_accuracy_panels(
    out_path: Path,
    *,
    props: List[Dict[str, Any]],
    prop_order: List[str],
    layers: List[str],
    mask_labels: List[str],
    title: str,
    vmin: float = 0.5,
    vmax: float = 1.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["baseline"] + mask_labels
    n_panels = len(panels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(10, 3.2 * n_panels), max(4.6, 0.65 * len(prop_order) + 2.0)),
        squeeze=False,
    )
    axes = axes[0]

    for ax, panel in zip(axes, panels):
        mat = _matrix_accuracy(props, prop_order, layers, panel=("baseline" if panel == "baseline" else panel))
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
        ax.set_yticks(range(len(prop_order)))
        ax.set_yticklabels(prop_order, fontsize=10)
        # Keep x labels readable: show ~12 ticks max.
        xs = np.arange(len(layers))
        step = max(1, len(layers) // 12)
        tick_idx = list(range(0, len(layers), step))
        if (len(layers) - 1) not in tick_idx:
            tick_idx.append(len(layers) - 1)
        tick_text = []
        for i in tick_idx:
            li = _layer_index(layers[i])
            tick_text.append(f"layer {li}" if li is not None else str(i))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_text, rotation=45, ha="right", fontsize=8)
        ax.set_title("Baseline\n(dense)" if panel == "baseline" else panel, fontsize=11, fontweight="bold")
        # Cell annotations (lightweight): only when small.
        if len(layers) <= 16:
            for pi in range(mat.shape[0]):
                for li in range(mat.shape[1]):
                    v = mat[pi, li]
                    if np.isfinite(v):
                        ax.text(li, pi, f"{v:.2f}", ha="center", va="center", fontsize=7)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.subplots_adjust(right=0.90, wspace=0.35)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("probe accuracy (test)", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_panels(
    out_path: Path,
    *,
    props: List[Dict[str, Any]],
    prop_order: List[str],
    layers: List[str],
    mask_labels: List[str],
    title: str,
    lim: float = 0.6,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(mask_labels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(10, 3.2 * n_panels), max(4.6, 0.65 * len(prop_order) + 2.0)),
        squeeze=False,
    )
    axes = axes[0]

    for ax, mask in zip(axes, mask_labels):
        mat = _matrix_delta(props, prop_order, layers, mask_label=mask)
        im = ax.imshow(mat, vmin=-lim, vmax=lim, cmap="RdBu", aspect="auto")
        ax.set_yticks(range(len(prop_order)))
        ax.set_yticklabels(prop_order, fontsize=10)
        xs = np.arange(len(layers))
        step = max(1, len(layers) // 12)
        tick_idx = list(range(0, len(layers), step))
        if (len(layers) - 1) not in tick_idx:
            tick_idx.append(len(layers) - 1)
        tick_text = []
        for i in tick_idx:
            li = _layer_index(layers[i])
            tick_text.append(f"layer {li}" if li is not None else str(i))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_text, rotation=45, ha="right", fontsize=8)
        ax.set_title(mask, fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.subplots_adjust(right=0.90, wspace=0.35)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Δ accuracy (masked − dense)", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot dense-vs-mask builtin probe heatmaps.")
    p.add_argument("--input-json", type=str, required=True, help="Path to dense_vs_mask_probes.json")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNGs (default: <suite-dir>/probe_plots)",
    )
    args = p.parse_args()

    inp = Path(args.input_json).resolve()
    data = _load(inp)
    props = data.get("properties") or []
    if not isinstance(props, list) or not props:
        raise SystemExit(f"No properties in {inp}")

    mask_labels = data.get("mask_labels") or []
    if not isinstance(mask_labels, list) or not mask_labels:
        raise SystemExit("dense_vs_mask_probes.json missing mask_labels")

    prop_order = _property_order(props)
    layers = _ordered_layers(props)

    # Default output location: alongside suite outputs if present.
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (inp.parent / "probe_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_accuracy_panels(
        out_dir / "dense_vs_mask_builtin_accuracy.png",
        props=props,
        prop_order=prop_order,
        layers=layers,
        mask_labels=mask_labels,
        title="Probing (builtin): dense baseline vs masked subnetworks",
    )
    _plot_delta_panels(
        out_dir / "dense_vs_mask_builtin_delta.png",
        props=props,
        prop_order=prop_order,
        layers=layers,
        mask_labels=mask_labels,
        title="Δ Probe accuracy (masked − dense) by layer and task",
    )


if __name__ == "__main__":
    main()


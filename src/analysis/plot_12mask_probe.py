#!/usr/bin/env python3
"""
Generate a single reference-style probe heatmap with 12 masks (6 DPO + 6 GRPO).

Layout:
  Top:    Baseline (dense) absolute accuracy heatmap
  Row 1:  Δ Cold-SNIP-DPO,   Δ Cold-CAV-DPO,   Δ Cold-Fisher-DPO
  Row 2:  Δ Warm-Fisher-DPO, Δ Warm-Magnitude-DPO, Δ Warm-Momentum-DPO
  Row 3:  Δ Cold-SNIP-GRPO,  Δ Cold-CAV-GRPO,  Δ Cold-Fisher-GRPO
  Row 4:  Δ Warm-Fisher-GRPO,Δ Warm-Magnitude-GRPO,Δ Warm-Momentum-GRPO

Usage:
    python src/analysis/plot_12mask_probe.py \
        --results_dir /path/to/probe_results_llama8b \
        --output probe_results_llama8b/summary/probe_12masks.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


PROPERTIES = ["syntax", "semantics", "factual", "math"]
PROP_LABELS = [
    "Syntax\n(BLiMP)",
    "Semantics\n(SST-2)",
    "Factual\n(AG News)",
    "Math\n(GSM8K)",
]

# Which comparison JSON contains which masks
# Maps: method_key -> comparison directory name
COMPARISONS = {
    "Cold-SNIP":      "cold_snip_dpo_vs_grpo",
    "Cold-CAV":       "cold_cav_dpo_vs_grpo",
    "Cold-Fisher":    "cold_fisher_dpo_vs_grpo",
    "Warm-Fisher":    "warm_fisher_dpo_vs_grpo",
    "Warm-Magnitude": "warm_magnitude_dpo_vs_grpo",
    "Warm-Momentum":  "warm_momentum_dpo_vs_grpo",
}

# Panel layout: (row, col) -> (method_key, algo)
# Row 0-1: DPO (cold then warm), Row 2-3: GRPO (cold then warm)
PANEL_LAYOUT = [
    # Row 0: DPO Cold
    [("Cold-SNIP", "DPO"),    ("Cold-CAV", "DPO"),       ("Cold-Fisher", "DPO")],
    # Row 1: DPO Warm
    [("Warm-Fisher", "DPO"),  ("Warm-Magnitude", "DPO"), ("Warm-Momentum", "DPO")],
    # Row 2: GRPO Cold
    [("Cold-SNIP", "GRPO"),   ("Cold-CAV", "GRPO"),      ("Cold-Fisher", "GRPO")],
    # Row 3: GRPO Warm
    [("Warm-Fisher", "GRPO"), ("Warm-Magnitude", "GRPO"),("Warm-Momentum", "GRPO")],
]

ROW_LABELS = [
    "DPO — Cold-start",
    "DPO — Warm-start",
    "GRPO — Cold-start",
    "GRPO — Warm-start",
]


def load_results(results_dir):
    """Load all probe_results.json files."""
    data = {}
    for subdir in sorted(Path(results_dir).iterdir()):
        jp = subdir / "probe_results.json"
        if jp.exists():
            with open(jp) as f:
                data[subdir.name] = json.load(f)
    return data


def extract_per_layer(prop_results, layer_keys):
    """Extract ordered per-layer values for each property.

    Compatible with both the legacy format (value=float) and the new
    diagnostics format (value={"test": float, "train": ..., ...}).
    """
    def _get(v):
        return v["test"] if isinstance(v, dict) else v
    return {p: [_get(prop_results[p][lk]) for lk in layer_keys] for p in PROPERTIES}


def find_config(data, keyword):
    """Find config label in data that matches keyword (case-insensitive)."""
    for cfg in data:
        cfg_c = cfg.replace("\n", " ").strip().lower()
        if keyword in cfg_c:
            return cfg
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or os.path.join(args.results_dir, "summary", "probe_12masks.png")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    all_data = load_results(args.results_dir)

    # Detect layer keys from first available result
    sample = next(iter(all_data.values()))
    sample_cfg = next(iter(sample.values()))
    layer_keys = list(sample_cfg[PROPERTIES[0]].keys())
    layer_indices = [int(k.split(".")[2]) for k in layer_keys]
    n_layers = len(layer_indices)

    # ── Extract baseline + all mask per-layer data ──
    # Use first available comparison's baseline (they should all be identical
    # after the probe_cache fix)
    first_comp = next(iter(COMPARISONS.values()))
    bl_cfg = find_config(all_data[first_comp], "baseline") or find_config(all_data[first_comp], "no mask")
    baseline = extract_per_layer(all_data[first_comp][bl_cfg], layer_keys)

    # Extract each mask
    mask_data = {}  # (method_key, algo) -> {prop: [layer_vals]}
    for method_key, comp_name in COMPARISONS.items():
        if comp_name not in all_data:
            print(f"WARNING: {comp_name} not found, skipping {method_key}")
            continue
        data = all_data[comp_name]
        for algo in ["DPO", "GRPO"]:
            cfg = find_config(data, algo.lower())
            if cfg is None:
                # Try finding by method+algo combo
                cfg = find_config(data, method_key.split("-")[-1].lower())
            if cfg and ("baseline" not in cfg.lower() and "no mask" not in cfg.lower()):
                mask_data[(method_key, algo)] = extract_per_layer(data[cfg], layer_keys)

    # ── Compute delta matrices ──
    bl_mat = np.array([[baseline[p][li] for li in range(n_layers)] for p in PROPERTIES])

    delta_mats = {}
    for key, mdata in mask_data.items():
        mask_mat = np.array([[mdata[p][li] for li in range(n_layers)] for p in PROPERTIES])
        delta_mats[key] = mask_mat - bl_mat

    # Global delta range for consistent colorbar
    all_deltas = np.concatenate([m.flatten() for m in delta_mats.values()])
    abs_max = max(0.05, np.nanmax(np.abs(all_deltas)))
    abs_max = np.ceil(abs_max * 10) / 10

    # ── Figure layout ──
    n_delta_rows = len(PANEL_LAYOUT)
    fig_height = 4.2 + n_delta_rows * 3.2
    fig = plt.figure(figsize=(19, fig_height))

    # Gridspec: baseline on top, then delta panels
    # Reserve top ~18% for baseline, rest for 4 rows of deltas
    baseline_h = 0.14
    gap = 0.03
    delta_h = (1.0 - baseline_h - gap - 0.08) / n_delta_rows  # 0.08 for suptitle

    # ── Baseline heatmap ──
    ax_bl = fig.add_axes([0.15, 1.0 - baseline_h - 0.02, 0.65, baseline_h - 0.02])

    cmap_abs = plt.cm.RdYlGn
    norm_abs = mcolors.Normalize(vmin=0.45, vmax=1.0)
    im_bl = ax_bl.imshow(bl_mat, cmap=cmap_abs, norm=norm_abs, aspect="auto")

    ax_bl.set_xticks(range(n_layers))
    ax_bl.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
    ax_bl.set_yticks(range(len(PROPERTIES)))
    ax_bl.set_yticklabels(PROP_LABELS, fontsize=10, fontweight="bold")

    for ri in range(len(PROPERTIES)):
        for ci in range(n_layers):
            v = bl_mat[ri, ci]
            tc = "white" if v < 0.6 else "black"
            ax_bl.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                       fontsize=8, color=tc, fontweight="bold")

    cb_pos = [0.82, 1.0 - baseline_h - 0.02, 0.015, baseline_h - 0.02]
    cb_ax = fig.add_axes(cb_pos)
    cb_bl = fig.colorbar(im_bl, cax=cb_ax)
    cb_bl.set_label("Probe Accuracy", fontsize=10)

    # ── Delta panels: 4 rows × 3 cols ──
    cmap_delta = plt.cm.RdBu
    norm_delta = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    panel_left = 0.06
    panel_right = 0.86
    panel_w = (panel_right - panel_left - 0.06) / 3  # 3 cols with gaps
    col_gap = 0.03

    delta_top = 1.0 - baseline_h - gap - 0.04
    row_gap = 0.06
    section_gap = 0.03  # extra gap between DPO and GRPO sections
    panel_h = (delta_top - 0.04 - row_gap * (n_delta_rows - 1) - section_gap) / n_delta_rows

    for ri, row_panels in enumerate(PANEL_LAYOUT):
        # Add extra gap between DPO (rows 0-1) and GRPO (rows 2-3)
        extra = section_gap if ri >= 2 else 0
        y = delta_top - (ri + 1) * panel_h - ri * row_gap - extra

        for ci, (method_key, algo) in enumerate(row_panels):
            x = panel_left + ci * (panel_w + col_gap)
            ax = fig.add_axes([x, y, panel_w, panel_h])

            key = (method_key, algo)
            if key in delta_mats:
                mat = delta_mats[key]
                ax.imshow(mat, cmap=cmap_delta, norm=norm_delta, aspect="auto")

                for pri in range(len(PROPERTIES)):
                    for li in range(n_layers):
                        v = mat[pri, li]
                        sign = "+" if v > 0 else ""
                        tc = "white" if abs(v) > abs_max * 0.5 else "black"
                        ax.text(li, pri, f"{sign}{v:.2f}", ha="center", va="center",
                                fontsize=7, color=tc, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")

            title = f"Δ {method_key}"
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([str(i) for i in layer_indices], fontsize=7)
            ax.set_yticks(range(len(PROPERTIES)))
            if ci == 0:
                ax.set_yticklabels(PROP_LABELS, fontsize=8)
            else:
                ax.set_yticklabels([])

        # Row label on the right side
        label_x = panel_right + 0.01
        label_y = y + panel_h / 2
        fig.text(label_x, label_y, ROW_LABELS[ri], fontsize=11, fontweight="bold",
                 rotation=270, ha="left", va="center", color="#333333")

    # Shared delta colorbar
    cbar_x = 0.91
    cbar_y = delta_top - n_delta_rows * panel_h - (n_delta_rows - 1) * row_gap - section_gap
    cbar_h = delta_top - cbar_y
    cbar_ax = fig.add_axes([cbar_x, cbar_y, 0.015, cbar_h])
    cb_d = fig.colorbar(plt.cm.ScalarMappable(norm=norm_delta, cmap=cmap_delta), cax=cbar_ax)
    cb_d.set_label("Δ Accuracy (masked − baseline)", fontsize=10)

    fig.suptitle(
        "Probing Classifier: 12 Pruning Masks (DPO + GRPO) — LLaMA-3.1-8B\n"
        "Knowledge Retention per Layer\n"
        "Baseline — Dense (no mask)",
        fontsize=14, fontweight="bold", y=1.005,
    )

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.close()


if __name__ == "__main__":
    main()

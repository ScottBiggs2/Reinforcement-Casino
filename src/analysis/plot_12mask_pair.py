#!/usr/bin/env python3
"""
12-mask summary plot for pairwise-probe results.

Takes a single merged JSON (13 configs: baseline + 12 masks) produced by
probe_pair_12masks.py and draws the same 4x3 delta-panel layout as
plot_12mask_probe.py, but from one file rather than six comparison dirs.

Usage:
    python src/analysis/plot_12mask_pair.py \\
        --results /scratch/xie.yiyi/probe_pair_merged_v2/probe_pair_results.json \\
        --output  /scratch/xie.yiyi/probe_pair_merged_v2/probe_pair_12masks.png
"""

import argparse
import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plot_12mask_probe import (
    PANEL_LAYOUT,
    PROPERTIES,
    PROP_LABELS,
    ROW_LABELS,
    extract_per_layer,
    find_config,
)


# Method-key + algo → label key in the merged pair JSON.
# plot_12mask_probe.py's find_config() does case-insensitive substring
# matching, so our merged labels ("Cold-SNIP-DPO" etc.) work as-is.
def _find_mask(data: dict, method_key: str, algo: str) -> str | None:
    """Locate the merged-JSON label for a given (method, algo) combo."""
    needle = f"{method_key}-{algo}".lower()
    for label in data:
        if label.replace("\n", " ").strip().lower() == needle:
            return label
    # Fallback: substring
    for label in data:
        if needle in label.lower():
            return label
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Merged probe_pair_results.json (baseline + 12 masks)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=100,
                        help="Figure DPI. 150 = publication quality but ~1GB RAM, "
                             "100 = default, 80 = fits in login-node memory limits.")
    parser.add_argument("--figscale", type=float, default=0.7,
                        help="Multiplier on figure size (default 0.7 = smaller, "
                             "1.0 = original publication size).")
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(args.results), "probe_pair_12masks.png"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    # ── Detect layer keys ───────────────────────────────────────────────
    sample_cfg = next(iter(data.values()))
    layer_keys = list(sample_cfg[PROPERTIES[0]].keys())
    layer_indices = [int(k.split(".")[2]) for k in layer_keys]
    n_layers = len(layer_indices)

    # ── Baseline ────────────────────────────────────────────────────────
    bl_label = find_config(data, "baseline") or find_config(data, "no mask")
    if bl_label is None:
        raise RuntimeError("No baseline config found in results JSON")
    baseline = extract_per_layer(data[bl_label], layer_keys)
    bl_mat = np.array([[baseline[p][li] for li in range(n_layers)] for p in PROPERTIES])

    # ── Mask configs → delta mats ──────────────────────────────────────
    mask_data = {}
    for row in PANEL_LAYOUT:
        for method_key, algo in row:
            label = _find_mask(data, method_key, algo)
            if label is None:
                print(f"WARNING: {method_key}-{algo} not found in JSON, will be N/A")
                continue
            mask_data[(method_key, algo)] = extract_per_layer(data[label], layer_keys)

    delta_mats = {}
    for key, mdata in mask_data.items():
        mask_mat = np.array([[mdata[p][li] for li in range(n_layers)] for p in PROPERTIES])
        delta_mats[key] = mask_mat - bl_mat

    all_deltas = np.concatenate([m.flatten() for m in delta_mats.values()])
    abs_max = max(0.05, np.nanmax(np.abs(all_deltas)))
    abs_max = np.ceil(abs_max * 10) / 10

    # ── Layout (same as plot_12mask_probe.py) ──────────────────────────
    n_delta_rows = len(PANEL_LAYOUT)
    fig_height = (4.2 + n_delta_rows * 3.2) * args.figscale
    fig_width = 19 * args.figscale
    fig = plt.figure(figsize=(fig_width, fig_height))

    baseline_h = 0.14
    gap = 0.03
    delta_top = 1.0 - baseline_h - gap - 0.04
    row_gap = 0.06
    section_gap = 0.03
    panel_h = (delta_top - 0.04 - row_gap * (n_delta_rows - 1) - section_gap) / n_delta_rows
    panel_left = 0.06
    panel_right = 0.86
    panel_w = (panel_right - panel_left - 0.06) / 3
    col_gap = 0.03

    # Baseline strip
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
    cb_ax = fig.add_axes([0.82, 1.0 - baseline_h - 0.02, 0.015, baseline_h - 0.02])
    cb_bl = fig.colorbar(im_bl, cax=cb_ax)
    cb_bl.set_label("Probe Accuracy (pairwise)", fontsize=10)

    # Delta panels
    cmap_delta = plt.cm.RdBu
    norm_delta = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    for ri, row_panels in enumerate(PANEL_LAYOUT):
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

            ax.set_title(f"Δ {method_key}", fontsize=10, fontweight="bold", pad=4)
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([str(i) for i in layer_indices], fontsize=7)
            ax.set_yticks(range(len(PROPERTIES)))
            if ci == 0:
                ax.set_yticklabels(PROP_LABELS, fontsize=8)
            else:
                ax.set_yticklabels([])

        fig.text(panel_right + 0.01, y + panel_h / 2, ROW_LABELS[ri],
                 fontsize=11, fontweight="bold",
                 rotation=270, ha="left", va="center", color="#333333")

    cbar_y = delta_top - n_delta_rows * panel_h - (n_delta_rows - 1) * row_gap - section_gap
    cbar_h = delta_top - cbar_y
    cbar_ax = fig.add_axes([0.91, cbar_y, 0.015, cbar_h])
    cb_d = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_delta, cmap=cmap_delta), cax=cbar_ax
    )
    cb_d.set_label("Δ Pairwise Accuracy (masked − baseline)", fontsize=10)

    fig.suptitle(
        "Pairwise-Ranking Probe: 12 Pruning Masks (DPO + GRPO) — LLaMA-3.1-8B\n"
        "Linear Direction w maximising log σ(w·h+ − w·h−) per Layer\n"
        "Baseline — Dense (no mask)",
        fontsize=14, fontweight="bold", y=1.005,
    )

    plt.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.close()


if __name__ == "__main__":
    main()

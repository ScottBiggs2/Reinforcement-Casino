#!/usr/bin/env python3
r"""Figures for src/analysis/certifiability_margins.py — the Figure 5 replacement.

Reads ``certifiability_margins.npz`` + ``certifiability_diagnostics.json`` from one or more analysis
directories and writes titled PDF/ECDF panels per model plus a combined cross-model figure.

Encoding decisions (dataviz method — form, then color, then marks):

* The k-sweep is **ordinal**, not categorical: swapping k=50 and k=200 would change the meaning, so
  it takes a single hue in monotone lightness steps (blue 350/450/550/700, validated with
  ``validate_palette.js --ordinal``: monotone L, adjacent dL >= 0.06, light end 2.91:1 on surface).
  Reading darker as "more training" is the point of the encoding.
* The **oracle** is a different entity, not a step of that ramp, so it gets categorical slot 2
  (orange ``#eb6834``): CVD dE 23.1 against the lightest ramp step, 25.5 against the darkest, both
  well clear of the >= 8 target.
* The **random** arm is a null baseline, so it is neutral ink (``#52514e``) and dashed — hue is
  reserved for things that carry method identity.
* Chrome is the reference instance's: solid hairline grid (never dashed — dashing reads as
  "threshold"), muted axis ink, no mark borders, legend always present with selective annotation
  rather than a label on every curve.

Type: one sans family for prose and math (``mathtext.fontset=dejavusans``) so the figure has a single
type system. If the paper wants Computer Modern math, pass ``--mathtext cm``.

No dark-mode variant: the output medium is a printed/PDF page, which has one surface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---- reference palette instance (see dataviz references/palette.md)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RAMP_BLUE = ["#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal, light -> dark
ORACLE = "#eb6834"
RANDOM = INK_SECONDARY

QUANTITY_META = {
    "margin_rel": {
        "xlabel": r"$\tilde m_i = |\,s_i - \hat\tau_\rho(s)\,|\;/\;\hat\tau_\rho(s)$",
        "short": r"$\tilde m_i$",
        "name": "certifiability margin (relative to the selection boundary)",
    },
    "margin_sel": {
        "xlabel": r"$m_i = |\,s_i - \hat\tau_\rho(s)\,|$",
        "short": r"$m_i$",
        "name": "certifiability margin on the selection score (tie-break included)",
    },
    "margin_raw": {
        "xlabel": r"$m_i = |\,s_i - \hat\tau_\rho(s)\,|$",
        "short": r"$m_i$",
        "name": "certifiability margin (raw score units)",
    },
    "gap_rel": {
        "xlabel": r"$\tilde\delta_i = |\,s_i/\hat\tau_\rho(s) - s^{*}_i/\hat\tau_\rho(s^{*})\,|$",
        "short": r"$\tilde\delta_i$",
        "name": "score gap to the oracle (relative)",
    },
    "margin_rel_sig_oracle": {
        "xlabel": r"$\tilde m_i = |\,s_i - \hat\tau_\rho(s)\,|\;/\;\hat\tau_\rho(s)$",
        "short": r"$\tilde m_i$",
        "name": "certifiability margin, restricted to weights that moved ($s^{*}_i>0$)",
    },
    "margin_rel_sig_self": {
        "xlabel": r"$\tilde m_i = |\,s_i - \hat\tau_\rho(s)\,|\;/\;\hat\tau_\rho(s)$",
        "short": r"$\tilde m_i$",
        "name": r"certifiability margin, restricted to each arm's own support ($s_i>0$)",
    },
    "gap_raw": {
        "xlabel": r"$\delta_i = |\,s_i - s^{*}_i\,|$",
        "short": r"$\delta_i$",
        "name": "score gap to the oracle (raw score units)",
    },
}


# ---------------------------------------------------------------- data access


class Cell:
    """One analysis directory: the arms, their histograms, and their diagnostics."""

    def __init__(self, path: Path, label: Optional[str] = None):
        self.path = path
        npz = path / "certifiability_margins.npz"
        diag_p = path / "certifiability_diagnostics.json"
        if not npz.is_file():
            raise FileNotFoundError(f"{npz} (run --execution_mode merge first)")
        if not diag_p.is_file():
            raise FileNotFoundError(f"{diag_p}")
        self.data = np.load(npz)
        self.diag = json.loads(diag_p.read_text(encoding="utf-8"))
        self.meta = self.diag.get("meta", {})
        self.label = label or self._auto_label()

    def _auto_label(self) -> str:
        parts = [self.meta.get("model_label") or "", self.meta.get("objective_label") or ""]
        ds = self.meta.get("dataset_label") or ""
        head = " ".join(p for p in parts if p).strip()
        if ds:
            head = f"{head} on {ds}" if head else ds
        return head or self.path.name

    def sparsities(self) -> List[float]:
        return sorted({float(x) for x in self.meta.get("sparsity_percents", [])})

    def arm_names(self) -> List[str]:
        seen: List[str] = []
        for tag in self.diag.get("arms", {}):
            arm = tag.rsplit("_rho", 1)[0]
            if arm not in seen:
                seen.append(arm)
        return seen

    def record(self, arm: str, rho: float) -> Optional[dict]:
        return self.diag.get("arms", {}).get(f"{arm}_rho{rho:g}")

    def series(self, arm: str, rho: float, quantity: str) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        tag = f"{arm}_rho{rho:g}"
        ck = f"{tag}_{quantity}_counts"
        if ck not in self.data.files:
            return None
        counts = self.data[ck]
        edges = self.data[f"{tag}_{quantity}_log_edges"]
        under = int(self.data[f"{tag}_{quantity}_underflow"][0])
        over = int(self.data[f"{tag}_{quantity}_overflow"][0])
        if counts.sum() + under + over <= 0:
            return None
        return counts, edges, under, over

    def degenerate_arms(self, rho: float) -> List[str]:
        out = []
        for arm in self.arm_names():
            rec = self.record(arm, rho)
            if rec and rec.get("tau", {}).get("tau_degenerate"):
                out.append(arm)
        return out


def arm_style(arm: str, milestones: Sequence[int]) -> Dict[str, object]:
    """Ordinal ramp over k; distinct hue for the oracle; neutral dashed for the null baseline."""
    if arm == "oracle":
        return {"color": ORACLE, "lw": 2.4, "ls": "-", "z": 5, "label": r"oracle  $s^{*}=|\theta^{T}-\theta^{0}|$"}
    if arm.startswith("random"):
        return {"color": RANDOM, "lw": 2.0, "ls": (0, (5, 2)), "z": 4, "label": "random baseline"}
    k = int(arm.split("_k")[-1])
    order = sorted(int(m) for m in milestones) or [k]
    i = order.index(k) if k in order else 0
    color = RAMP_BLUE[min(i, len(RAMP_BLUE) - 1)] if len(order) <= len(RAMP_BLUE) else _ramp_at(i, len(order))
    return {"color": color, "lw": 2.0, "ls": "-", "z": 3, "label": rf"warm magnitude, $k={k}$"}


def _ramp_at(i: int, n: int) -> str:
    """Interpolate within the validated blue ramp if a sweep has more than four milestones."""
    if n <= 1:
        return RAMP_BLUE[-1]
    pos = i / (n - 1) * (len(RAMP_BLUE) - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if lo == hi:
        return RAMP_BLUE[lo]
    t = pos - lo
    c0 = np.array(matplotlib.colors.to_rgb(RAMP_BLUE[lo]))
    c1 = np.array(matplotlib.colors.to_rgb(RAMP_BLUE[hi]))
    return matplotlib.colors.to_hex(tuple(c0 * (1 - t) + c1 * t))


def _spike_mask(log_edges: np.ndarray) -> np.ndarray:
    """Bins holding m~ = 1 exactly. Which side of the bin edge log10(1)=0 falls on is float-rounding
    dependent, so take both bins touching 0."""
    width = float(log_edges[1] - log_edges[0])
    centers = (log_edges[:-1] + log_edges[1:]) / 2.0
    return np.abs(centers) <= width


def _arm_sort_key(arm: str) -> Tuple[int, int]:
    if arm.startswith("warm_k"):
        return (0, int(arm.split("_k")[-1]))
    if arm == "oracle":
        return (1, 0)
    return (2, 0)


# ---------------------------------------------------------------- curve math


def density(counts: np.ndarray, log_edges: np.ndarray, total: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mass per decade, dP/dlog10(x) — the density that belongs on a log axis.

    Dividing by the *linear* bin width instead (what the old plotting script did) makes the ordinate
    blow up as x -> 0, because log-spaced bins get exponentially narrow: on these distributions it
    produced a ~1e11 axis offset and flattened every curve to zero.
    """
    if total <= 0:
        return np.array([]), np.array([])
    centers = 10 ** ((log_edges[:-1] + log_edges[1:]) / 2.0)
    widths = np.clip(log_edges[1:] - log_edges[:-1], 1e-12, None)
    return centers, counts.astype(np.float64) / total / widths


def ecdf(counts: np.ndarray, log_edges: np.ndarray, under: int, total: float) -> Tuple[np.ndarray, np.ndarray]:
    """Underflow (including exact zeros) is real mass at the left, so it seeds the cumulative sum."""
    if total <= 0:
        return np.array([]), np.array([])
    centers = 10 ** ((log_edges[:-1] + log_edges[1:]) / 2.0)
    cum = (under + np.cumsum(counts.astype(np.float64))) / total
    return centers, cum


def xlim_for(
    series: Sequence[Tuple[np.ndarray, np.ndarray]],
    mass: float,
    floor: float,
    low_mass: float = 1e-4,
) -> Tuple[float, float]:
    """Union range over the plotted series, quantile-trimmed at BOTH ends.

    The low end is taken from a quantile rather than the first occupied bin: a handful of
    coordinates where s_i happens to land on tau put isolated counts many decades below the bulk,
    and anchoring to them collapses the informative region. It must not be a fixed floor either --
    a hard 1e-14 clamp cut off the s_i = 0 mass entirely, which sits at m_i = tau and is the
    dominant feature of these distributions.
    """
    lo, hi = np.inf, 0.0
    for counts, log_edges in series:
        total = float(counts.sum())
        if total <= 0:
            continue
        nz = np.flatnonzero(counts > 0)
        if nz.size == 0:
            continue
        cum = np.cumsum(counts.astype(np.float64)) / total
        j = min(int(np.searchsorted(cum, mass)), len(counts) - 1)
        hi = max(hi, float(10 ** log_edges[j + 1]) * 1.15)
        jl = min(int(np.searchsorted(cum, low_mass)), len(counts) - 1)
        lo_j = min(jl, int(nz[0]))
        lo = min(lo, float(10 ** log_edges[lo_j]) * 0.85)
    if hi <= 0:
        return floor, 1.0
    return max(lo, floor), hi


# ---------------------------------------------------------------- drawing


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    # Solid hairline grid; dashed grid reads as a threshold line.
    ax.grid(True, which="major", ls="-", lw=0.6, color=GRID, zorder=0)
    ax.grid(True, which="minor", ls="-", lw=0.4, color=GRID, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, which="both", labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_SECONDARY)


def draw_panel(
    ax,
    cell: Cell,
    rho: float,
    quantity: str,
    kind: str,
    *,
    milestones: Sequence[int],
    annotate_spike: bool,
) -> int:
    drawn = 0
    plotted: List[Tuple[np.ndarray, np.ndarray]] = []
    continuum_max = 0.0
    spike_clipped = False
    for arm in sorted(cell.arm_names(), key=_arm_sort_key):
        s = cell.series(arm, rho, quantity)
        if s is None:
            continue
        counts, log_edges, under, over = s
        total = float(counts.sum() + under + over)
        st = arm_style(arm, milestones)
        if kind == "pdf":
            x, y = density(counts, log_edges, total)
            ax.plot(x, y, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"], label=st["label"])
            if quantity == "margin_rel" and y.size:
                # The s_i == 0 mass is a single spike at m~ = 1; on the real runs it is ~99% of the
                # coordinates and would flatten every other feature to the axis. Scale to the
                # continuum and let the ECDF panel carry the spike's magnitude (its final jump).
                off = _spike_mask(log_edges)
                cm_ = float(y[~off].max()) if (~off).any() else 0.0
                continuum_max = max(continuum_max, cm_)
                if off.any() and float(y[off].max()) > 1.5 * cm_:
                    spike_clipped = True
        else:
            x, y = ecdf(counts, log_edges, under, total)
            ax.step(x, y, where="mid", color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"], label=st["label"])
        plotted.append((counts, log_edges))
        drawn += 1

    if not drawn:
        ax.text(
            0.5,
            0.5,
            "no non-degenerate arm at this $\\rho$",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=INK_MUTED,
            fontsize=13,
        )
        style_axes(ax)
        return 0

    lo, hi = xlim_for(plotted, mass=0.9995, floor=1e-40, low_mass=args_low_mass())
    ax.set_xscale("log")
    ax.set_xlim(_XLIM[0] if _XLIM[0] else lo, _XLIM[1] if _XLIM[1] else hi)
    style_axes(ax)

    # Selective annotation instead of labelling every curve: for tau-relative margins the value 1
    # is exactly "score is zero", so the spike there is the share of the mask with no signal.
    if annotate_spike and quantity == "margin_rel" and lo <= 1.0 <= hi:
        ax.axvline(1.0, color=INK_MUTED, lw=0.9, zorder=1)
        ax.annotate(
            r"$s_i=0$",
            xy=(1.0, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(-5, -6),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=10,
            color=INK_SECONDARY,
        )

    if kind == "pdf":
        ax.set_ylabel(r"$dP\,/\,d\log_{10}$", color=INK_SECONDARY)
        if spike_clipped and continuum_max > 0:
            ax.set_ylim(0, continuum_max * 1.20)
            ax.annotate(
                r"spike at $\tilde m_i=1$ clipped" "\n" r"(height = ECDF jump at right)",
                xy=(1.0, 0.90),
                xycoords=("data", "axes fraction"),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=10,
                color=INK_SECONDARY,
            )
    else:
        ax.set_ylim(0, 1)
        ax.set_ylabel(rf"$\Pr[\,${QUANTITY_META[quantity]['short']}$\,\leq x\,]$", color=INK_SECONDARY)
    ax.set_xlabel(QUANTITY_META[quantity]["xlabel"], color=INK_SECONDARY)
    return drawn


def subtitle_for(cell: Cell, rho: float, quantity: str) -> str:
    bits = [QUANTITY_META[quantity]["name"]]
    if quantity == "margin_rel":
        bits.append(r"$\tilde m_i=1$ marks $s_i=0$ (no resolvable displacement)")
    if quantity.startswith("margin_rel_sig"):
        shares = [
            rec.get("frac_score_exactly_zero")
            for arm in cell.arm_names()
            for rec in [cell.record(arm, rho)]
            if rec and str(arm).startswith(("warm", "oracle"))
        ]
        shares = [x for x in shares if isinstance(x, (int, float))]
        if shares:
            bits.append(
                rf"excludes the $s_i=0$ mass ({min(shares):.4f}--{max(shares):.4f} of coordinates)"
            )
    scope = cell.meta.get("scope") or {}
    if scope.get("n_tensors"):
        bits.append(f"{scope['n_tensors']} tensors / {int(scope.get('covered_elements', 0)):,} weights")
    rule = cell.meta.get("tau_rule")
    if rule == "hybrid_global_phase":
        bits.append(
            rf"$\hat\tau_\rho$ from the hybrid global selection (floor "
            rf"{cell.meta.get('hybrid_min_layer_keep_ratio')})"
        )
    # Only the tau-normalized quantities are gated on the flag; margin_raw plots every arm.
    if quantity.endswith("_rel") or quantity.startswith("margin_rel_sig"):
        deg = cell.degenerate_arms(rho)
        if deg:
            bits.append("omitted as degenerate: " + ", ".join(deg))
    return "  ·  ".join(str(b) for b in bits)


def rc(mathtext: str) -> Dict[str, object]:
    return {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "mathtext.fontset": mathtext,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "axes.edgecolor": AXIS,
        "text.color": INK_PRIMARY,
    }


def legend_on(ax) -> None:
    leg = ax.legend(loc="best", frameon=True, framealpha=0.94, edgecolor=GRID, facecolor=SURFACE, borderpad=0.7)
    leg.get_frame().set_linewidth(0.8)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)


def figure_per_cell(cell: Cell, rho: float, quantity: str, out_dir: Path, args) -> List[Path]:
    milestones = cell.meta.get("milestones", [])
    written: List[Path] = []
    with plt.rc_context(rc(args.mathtext)):
        fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.6))
        n_pdf = draw_panel(axes[0], cell, rho, quantity, "pdf", milestones=milestones, annotate_spike=True)
        draw_panel(axes[1], cell, rho, quantity, "ecdf", milestones=milestones, annotate_spike=True)
        axes[0].set_title("Distribution", color=INK_PRIMARY, loc="left", pad=10)
        axes[1].set_title("Cumulative", color=INK_PRIMARY, loc="left", pad=10)
        if n_pdf:
            legend_on(axes[1])

        title = args.title or (
            rf"Certifiability margins at $\rho={rho:g}\%$" + (f" — {cell.label}" if cell.label else "")
        )
        fig.suptitle(title, fontsize=18, color=INK_PRIMARY, x=0.012, ha="left", y=0.985)
        fig.text(0.012, 0.925, subtitle_for(cell, rho, quantity), fontsize=11, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        stem = f"certifiability_{quantity}_rho{rho:g}"
        for ext in ("png", "pdf"):
            p = out_dir / f"{stem}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            written.append(p)
        plt.close(fig)
    return written


def figure_combined(cells: Sequence[Cell], rho: float, quantity: str, out_dir: Path, args) -> List[Path]:
    written: List[Path] = []
    with plt.rc_context(rc(args.mathtext)):
        fig, axes = plt.subplots(len(cells), 2, figsize=(16.0, 5.9 * len(cells)), squeeze=False)
        for r, cell in enumerate(cells):
            ms = cell.meta.get("milestones", [])
            n = draw_panel(axes[r][0], cell, rho, quantity, "pdf", milestones=ms, annotate_spike=(r == 0))
            draw_panel(axes[r][1], cell, rho, quantity, "ecdf", milestones=ms, annotate_spike=(r == 0))
            axes[r][0].set_title(f"{cell.label} — distribution", color=INK_PRIMARY, loc="left", pad=10)
            axes[r][1].set_title(f"{cell.label} — cumulative", color=INK_PRIMARY, loc="left", pad=10)
            if r == 0 and n:
                legend_on(axes[r][1])
            note = cell.degenerate_arms(rho) if quantity.endswith("_rel") or quantity.startswith("margin_rel_sig") else []
            if note:
                axes[r][0].text(
                    0.0,
                    -0.30,
                    "omitted as degenerate: " + ", ".join(note),
                    transform=axes[r][0].transAxes,
                    fontsize=10,
                    color=INK_MUTED,
                )
        # One shared x-range across rows: the whole point of stacking models is to compare them, and
        # per-row autoscaling would put the same m~ at different screen positions.
        drawn_axes = [a for row in axes for a in row if a.lines]
        if drawn_axes:
            lo = min(a.get_xlim()[0] for a in drawn_axes)
            hi = max(a.get_xlim()[1] for a in drawn_axes)
            for a in drawn_axes:
                a.set_xlim(lo, hi)

        title = args.title or rf"Certifiability margins at $\rho={rho:g}\%$ across models"
        fig.suptitle(title, fontsize=18, color=INK_PRIMARY, x=0.012, ha="left", y=0.995)
        fig.text(
            0.012,
            0.965,
            QUANTITY_META[quantity]["name"] + r"  ·  warm-magnitude $s(\theta^{\leq k})$ vs. oracle",
            fontsize=11,
            color=INK_MUTED,
            ha="left",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
        stem = f"certifiability_{quantity}_rho{rho:g}_combined"
        for ext in ("png", "pdf"):
            p = out_dir / f"{stem}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            written.append(p)
        plt.close(fig)
    return written


def write_table_view(cells: Sequence[Cell], rho: float, out_dir: Path) -> Path:
    """Non-visual path to the same numbers (accessibility + what the paper text quotes)."""
    lines = [
        f"# Certifiability margins at rho = {rho:g}%",
        "",
        "| model | arm | tau_hat | degenerate | signal support | keep budget | "
        "share of selection from tie-break | share of nonzero captured | s_i == 0 share | cert P[delta<m] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        for arm in sorted(cell.arm_names(), key=_arm_sort_key):
            rec = cell.record(arm, rho)
            if not rec:
                continue
            t = rec["tau"]
            lines.append(
                "| {m} | {a} | {tau:.4e} | {deg} | {sup:,} | {keep:,} | {tb} | {cap} | {zs} | {cert} |".format(
                    m=cell.label,
                    a=arm,
                    tau=float(t["tau"]),
                    deg="yes" if t["tau_degenerate"] else "no",
                    sup=int(t["n_raw_positive"]),
                    keep=int(t["keep_count"]),
                    tb=_fmt(rec.get("frac_of_selection_from_tie_break")),
                    cap=_fmt(rec.get("frac_of_nonzero_captured")),
                    zs=_fmt(rec.get("frac_score_exactly_zero")),
                    cert=_fmt(rec.get("cert_strict_fraction")),
                )
            )
    p = out_dir / f"certifiability_table_rho{rho:g}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if f != f else f"{f:.4f}"


# ---------------------------------------------------------------- cli


_LOW_MASS = [1e-4]
_XLIM = [None, None]


def args_low_mass() -> float:
    return _LOW_MASS[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Certifiability-margin figures.")
    ap.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Analysis dir, optionally labelled. Repeat for the combined cross-model figure.",
    )
    ap.add_argument("--analysis-dir", default=None, help="Shorthand for a single unlabelled --cell.")
    ap.add_argument("--out-dir", default=None, help="Default: <first cell>/figures")
    ap.add_argument("--rho", type=float, default=None, help="Sparsity to plot; default = all present.")
    ap.add_argument(
        "--quantity",
        action="append",
        default=[],
        choices=sorted(QUANTITY_META),
        help="Default: margin_rel and margin_raw.",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--mathtext", default="dejavusans", choices=("dejavusans", "cm", "stix"))
    ap.add_argument("--xmin", type=float, default=None, help="Override the left x limit.")
    ap.add_argument("--xmax", type=float, default=None, help="Override the right x limit.")
    ap.add_argument("--low-mass", type=float, default=1e-4, help="Quantile trim at the low end.")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    specs: List[Tuple[Optional[str], Path]] = []
    if args.analysis_dir:
        specs.append((None, Path(args.analysis_dir)))
    for c in args.cell:
        if "=" in c:
            label, d = c.split("=", 1)
            specs.append((label, Path(d)))
        else:
            specs.append((None, Path(c)))
    if not specs:
        ap.error("pass --analysis-dir or at least one --cell")

    _LOW_MASS[0] = float(args.low_mass)
    _XLIM[0], _XLIM[1] = args.xmin, args.xmax
    cells = [Cell(d, label) for label, d in specs]
    out_dir = Path(args.out_dir) if args.out_dir else cells[0].path / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    quantities = args.quantity or ["margin_sel", "margin_rel_sig_oracle", "margin_rel", "margin_raw"]
    rhos = [args.rho] if args.rho is not None else sorted({r for c in cells for r in c.sparsities()})
    if not rhos:
        raise SystemExit("No sparsity values found in the analysis metadata.")

    written: List[Path] = []
    for rho in rhos:
        for q in quantities:
            for cell in cells:
                sub = out_dir / _slug(cell.label) if len(cells) > 1 else out_dir
                sub.mkdir(parents=True, exist_ok=True)
                written += figure_per_cell(cell, rho, q, sub, args)
            if len(cells) > 1 and not args.no_combined:
                written += figure_combined(cells, rho, q, out_dir, args)
        written.append(write_table_view(cells, rho, out_dir))

    for p in written:
        print(f"wrote {p}")


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower() or "cell"


if __name__ == "__main__":
    main()

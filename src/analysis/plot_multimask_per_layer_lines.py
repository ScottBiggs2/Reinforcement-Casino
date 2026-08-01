#!/usr/bin/env python
"""Per-layer Jaccard + CKA line plot across mask pairs (reproduces the reference
multi_mask_per_layer_lines figure) from a `multi_mask_jaccard_cka.json`
produced by src/cold_start/mask_interpretation_suite.py.

Two panels side by side:
  left  = Jaccard per decoder layer (0..L-1), one line per selected pair
  right = linear CKA per down_proj layer, one line per selected pair

Selection: by default, every pair that involves --anchor (e.g. oracle_gt) is
drawn, labelled by the *other* mask in the pair. This gives the exact chart we
want: "CKA (and Jaccard) of each step-T magnitude mask vs the oracle, by layer."
Use --pairs to draw an explicit set instead.

Usage:
  python src/analysis/plot_multimask_per_layer_lines.py \
      --json  .../multi_mask_jaccard_cka.json \
      --anchor oracle_gt \
      --output .../multimask_per_layer_lines_vs_oracle.png \
      [--also-include "oracle_gt ⇄ random"] [--title "..."]
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEP = "⇄"


def _layer_idx(name: str) -> Optional[int]:
    m = re.search(r"layers?[._](\d+)", name)
    return int(m.group(1)) if m else None


def _parse_pair_key(key: str) -> Tuple[str, str]:
    a, b = key.split(f" {SEP} ") if f" {SEP} " in key else key.split(SEP)
    return a.strip(), b.strip()


def _cka_per_layer(entry: dict) -> Dict[int, float]:
    """down_proj-name -> value  ==>  layer_index -> value (drop None/NaN)."""
    out = {}
    for name, v in (entry.get("per_layer") or {}).items():
        if v is None:
            continue
        idx = _layer_idx(name)
        if idx is not None:
            out[idx] = float(v)
    return out


def _jaccard_per_layer(entry: dict) -> Dict[int, float]:
    """by_decoder_layer {"0": {"aggregate_jaccard": ..}} -> layer_index -> value."""
    out = {}
    bdl = entry.get("by_decoder_layer") or {}
    for k, v in bdl.items():
        if not str(k).isdigit():
            continue
        val = (v or {}).get("aggregate_jaccard")
        if val is not None:
            out[int(k)] = float(val)
    return out


def _resolve_pairs(jac: dict, cka: dict, anchor: Optional[str],
                   explicit: Optional[List[str]]) -> List[Tuple[str, str, str]]:
    """Return [(jaccard_key_or_None, cka_key_or_None, label), ...]."""
    all_keys = set(jac) | set(cka)
    result = []
    if explicit:
        for want in explicit:
            wa, wb = _parse_pair_key(want)
            match = None
            for k in all_keys:
                ka, kb = _parse_pair_key(k)
                if {ka, kb} == {wa, wb}:
                    match = k
                    break
            if match:
                result.append((match if match in jac else None,
                               match if match in cka else None, match))
        return result
    # anchor mode: every pair containing anchor, labelled by the other side
    for k in sorted(all_keys):
        ka, kb = _parse_pair_key(k)
        if anchor in (ka, kb):
            other = kb if ka == anchor else ka
            label = f"{other} vs {anchor}"
            result.append((k if k in jac else None, k if k in cka else None, label))
    return result


def _panel(ax, series, metric_getter, ylabel, title):
    drew = False
    for jkey, ckey, label, entry_src in series:
        d = metric_getter(jkey, ckey)
        if not d:
            continue
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], marker="o", ms=3, lw=1.4, label=label)
        drew = True
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    if drew:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="multi_mask_jaccard_cka.json")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--anchor", default="oracle_gt",
                    help="Draw every pair containing this label (default oracle_gt).")
    ap.add_argument("--pairs", default=None,
                    help="Comma-separated explicit pairs, e.g. 'mag_step50 ⇄ oracle_gt,...'. "
                         "Overrides --anchor.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    jac = data.get("jaccard", {}) or {}
    cka = data.get("cka", {}) or {}

    explicit = [p.strip() for p in args.pairs.split(",")] if args.pairs else None
    pairs = _resolve_pairs(jac, cka, args.anchor, explicit)
    if not pairs:
        raise SystemExit(f"No matching pairs (anchor={args.anchor!r}, pairs={explicit}). "
                         f"Available keys: {sorted(set(jac) | set(cka))}")

    # attach per-layer dicts
    j_series = [(jk, ck, lab, None) for (jk, ck, lab) in pairs]

    fig, (axj, axc) = plt.subplots(1, 2, figsize=(15, 6))
    _panel(axj, j_series,
           lambda jk, ck: _jaccard_per_layer(jac[jk]) if jk else {},
           "Jaccard", "Jaccard per layer")
    _panel(axc, j_series,
           lambda jk, ck: _cka_per_layer(cka[ck]) if ck else {},
           "CKA", "CKA per layer (HSIC linear CKA)")

    title = args.title or (f"Per-layer mask comparison vs {args.anchor} "
                           f"(Llama-3.1-8B, GRPO/OpenR1, 97.5% sparsity)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")

    # console dump of per-pair mean CKA/Jaccard for sanity
    for jk, ck, lab, _ in j_series:
        cd = _cka_per_layer(cka[ck]) if ck else {}
        jd = _jaccard_per_layer(jac[jk]) if jk else {}
        cm = round(sum(cd.values()) / len(cd), 4) if cd else None
        jm = round(sum(jd.values()) / len(jd), 4) if jd else None
        print(f"  {lab:32s} mean_CKA={cm}  mean_Jaccard={jm}  (cka_layers={len(cd)})")


if __name__ == "__main__":
    main()

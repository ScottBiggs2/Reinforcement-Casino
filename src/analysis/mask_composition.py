"""What is actually inside the subnetwork? Budget composition by tensor class.

WHY
---
`src/warm_start/checkpoint_diff_mask_finder.py` applies no exclusion list: LayerNorm
scales, biases, embeddings and lm_head are all scoreable. LayerNorm and bias
parameters move with large *relative* magnitude during fine-tuning, so any |dtheta|
scorer will over-select them. If the 2.5% subnetwork is largely LayerNorm + bias,
then:

  * BitFit (Ben Zaken et al., ACL 2022) already gets most of full fine-tuning on GLUE
    from ~0.08% of parameters -- biases only. The result would be a known one.
  * The random baseline also captures those tensors, via the per-layer keep floor
    (`min_layer_keep_ratio`, default 2.5e-3), which would EXPLAIN oracle ~= random --
    the single most damaging pattern in the reviews.
  * The LoRA comparison is unfair in our favour, since LoRA adapts only the 7 linear
    projections and structurally cannot touch LayerNorm.

None of that is knowable without counting. This script counts. CPU-only, seconds.

Reports per tensor class: share of all parameters, share of the retained budget, and
retention rate. The number that matters is the RATIO of the last two to the first --
the over-selection factor. A class with 0.01% of parameters holding 5% of the budget
is a 500x over-selection and is what the subnetwork is really made of.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, Tuple

import torch

# Ordered: first match wins, so put specific patterns before generic ones.
CLASS_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("bias", r"\.bias$"),
    ("embed", r"embed_tokens"),
    ("lm_head", r"lm_head"),
    # Qwen3 has per-head q_norm/k_norm in addition to the usual two per block
    ("layernorm", r"(layernorm|layer_norm|_norm|\bnorm\b)"),
    ("attn_qkv", r"\.(q_proj|k_proj|v_proj)\."),
    ("attn_o", r"\.o_proj\."),
    ("mlp_gate_up", r"\.(gate_proj|up_proj)\."),
    ("mlp_down", r"\.down_proj\."),
)


def classify_tensor(name: str) -> str:
    for cls, pat in CLASS_PATTERNS:
        if re.search(pat, name):
            return cls
    return "other"


def load_mask(path: str) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "masks" in obj:
        obj = obj["masks"]
    return obj


def audit(path: str) -> Dict[str, object]:
    masks = load_mask(path)
    total = defaultdict(int)
    kept = defaultdict(int)
    n_tensors = defaultdict(int)

    for name, m in masks.items():
        if not torch.is_tensor(m):
            continue
        cls = classify_tensor(name)
        total[cls] += m.numel()
        kept[cls] += int(m.sum().item()) if m.dtype == torch.bool else int((m != 0).sum().item())
        n_tensors[cls] += 1

    all_total = sum(total.values())
    all_kept = sum(kept.values())

    rows = []
    for cls in sorted(total, key=lambda c: -kept[c]):
        p_share = total[cls] / all_total if all_total else 0.0
        b_share = kept[cls] / all_kept if all_kept else 0.0
        rows.append({
            "class": cls,
            "tensors": n_tensors[cls],
            "params": total[cls],
            "param_share": p_share,
            "kept": kept[cls],
            "budget_share": b_share,
            "retention": kept[cls] / total[cls] if total[cls] else 0.0,
            # >1 means this class is over-represented in the subnetwork relative to
            # its share of the model. This is the column to read.
            "over_selection": (b_share / p_share) if p_share else float("inf"),
        })

    return {
        "mask": path,
        "total_params": all_total,
        "kept_params": all_kept,
        "overall_density": all_kept / all_total if all_total else 0.0,
        "overall_sparsity": 1 - (all_kept / all_total) if all_total else 0.0,
        "by_class": rows,
    }


def print_report(rep: Dict[str, object]) -> None:
    print(f"\n{'=' * 100}")
    print(f"MASK: {rep['mask']}")
    print(f"  kept {rep['kept_params']:,} / {rep['total_params']:,} "
          f"= density {rep['overall_density']*100:.4f}%  (sparsity {rep['overall_sparsity']*100:.4f}%)")
    print(f"{'=' * 100}")
    print(f"{'class':<14}{'tensors':>8}{'params':>16}{'% of model':>12}"
          f"{'kept':>14}{'% of budget':>13}{'retention':>11}{'over-sel':>10}")
    print("-" * 100)
    for r in rep["by_class"]:
        print(f"{r['class']:<14}{r['tensors']:>8}{r['params']:>16,}{r['param_share']*100:>11.4f}%"
              f"{r['kept']:>14,}{r['budget_share']*100:>12.4f}%"
              f"{r['retention']*100:>10.4f}%{r['over_selection']:>10.2f}x")
    print("-" * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tensor-class composition of a subnetwork mask")
    ap.add_argument("masks", nargs="+", help="One or more mask .pt files (oracle, random, ...)")
    ap.add_argument("--out", default=None, help="Write JSON report here")
    args = ap.parse_args()

    reports = []
    for p in args.masks:
        rep = audit(p)
        print_report(rep)
        reports.append(rep)

    if len(reports) > 1:
        # The comparison that matters: does the scored mask concentrate the budget in
        # classes the random baseline also happens to cover (via the per-layer floor)?
        # If the budget-share columns match, "oracle vs random" is not testing scoring.
        classes = sorted({r["class"] for rep in reports for r in rep["by_class"]})
        print(f"\n{'=' * 100}")
        print("BUDGET SHARE BY CLASS — if these columns agree, oracle and random are")
        print("spending their budget on the same kinds of tensors.")
        print(f"{'=' * 100}")
        names = [rep["mask"].split("/")[-1][:26] for rep in reports]
        print(f"{'class':<14}" + "".join(f"{n:>28}" for n in names))
        print("-" * 100)
        for cls in classes:
            line = f"{cls:<14}"
            for rep in reports:
                hit = next((r for r in rep["by_class"] if r["class"] == cls), None)
                line += f"{(hit['budget_share']*100 if hit else 0.0):>27.4f}%"
            print(line)
        print("-" * 100)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(reports, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

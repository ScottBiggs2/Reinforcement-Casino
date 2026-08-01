"""
Gate a freshly built mask against a reference mask that is already known-good.

src/utils/verify_mask_coverage.py answers "do these keys and shapes match the model?" and needs a
full fp32 model load to do it. When a reference mask for the same model already exists and has
been trained on, comparing against that reference is both cheaper and stricter: it catches key-set
drift, shape drift, a wrong sparsity, a missing per-layer floor, and a degenerate all-zero tensor,
without touching the model at all.

Also reports mask-vs-reference Jaccard, which is the quantity of interest anyway when the two
masks are warm-start selections at different milestones k.

  python scripts/verify_mask_against_reference.py \
      --reference .../warm_magnitude_..._step250.pt \
      --mask .../warm_magnitude_..._step50.pt \
      --mask .../warm_magnitude_..._step100.pt \
      --expect_keep_rate 0.025 --expect_min_layer_keep_ratio 0.0025 \
      --out_json .../mask_gate.json

Exits non-zero if any check fails.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.utils.mask_utils import load_masks_file


def summarize(path):
    masks, meta, wrapped = load_masks_file(path)
    per = {}
    for name, m in masks.items():
        mb = m.bool()
        per[name] = {
            "shape": tuple(mb.shape),
            "n": int(mb.numel()),
            "k": int(mb.sum().item()),
        }
    total_n = sum(v["n"] for v in per.values())
    total_k = sum(v["k"] for v in per.values())
    rates = {k: v["k"] / v["n"] for k, v in per.items()}
    return {
        "path": path,
        "wrapped_format": wrapped,
        "metadata": meta or {},
        "n_tensors": len(per),
        "n_elements": total_n,
        "k_kept": total_k,
        "keep_rate": total_k / total_n if total_n else None,
        "min_tensor_keep_rate": min(rates.values()) if rates else None,
        "max_tensor_keep_rate": max(rates.values()) if rates else None,
        "n_empty_tensors": sum(1 for v in per.values() if v["k"] == 0),
        "_per": per,
    }, masks


def jaccard(a, b):
    inter = union = 0
    for name in a:
        x, y = a[name].bool(), b[name].bool()
        inter += int((x & y).sum().item())
        union += int((x | y).sum().item())
    return inter / union if union else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--mask", action="append", required=True)
    p.add_argument("--expect_keep_rate", type=float, default=None,
                   help="Target 1-rho; checked to within --keep_rate_tol.")
    p.add_argument("--keep_rate_tol", type=float, default=0.001)
    p.add_argument("--expect_min_layer_keep_ratio", type=float, default=None,
                   help="Must appear in mask metadata, and no tensor may fall below it.")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    print(f"reference: {args.reference}")
    ref_sum, ref_masks = summarize(args.reference)
    print(f"  tensors={ref_sum['n_tensors']}  elements={ref_sum['n_elements']:,}  "
          f"keep={ref_sum['keep_rate']:.6f}")

    failures, out = [], {"reference": {k: v for k, v in ref_sum.items() if k != "_per"}, "masks": []}

    for path in args.mask:
        print(f"\nmask: {path}")
        s, masks = summarize(path)
        checks = []

        if set(masks) != set(ref_masks):
            only_m = sorted(set(masks) - set(ref_masks))[:5]
            only_r = sorted(set(ref_masks) - set(masks))[:5]
            checks.append(("key_set_matches_reference", False,
                           f"only-in-mask={only_m} only-in-reference={only_r}"))
        else:
            checks.append(("key_set_matches_reference", True, f"{len(masks)} keys"))
            bad_shapes = [
                (n, s["_per"][n]["shape"], ref_sum["_per"][n]["shape"])
                for n in masks if s["_per"][n]["shape"] != ref_sum["_per"][n]["shape"]
            ]
            checks.append(("shapes_match_reference", not bad_shapes, str(bad_shapes[:3])))

        if args.expect_keep_rate is not None:
            ok = abs(s["keep_rate"] - args.expect_keep_rate) <= args.keep_rate_tol
            checks.append(("global_keep_rate", ok,
                           f"{s['keep_rate']:.6f} vs expected {args.expect_keep_rate}"))

        mlkr = s["metadata"].get("min_layer_keep_ratio")
        if args.expect_min_layer_keep_ratio is not None:
            checks.append(("metadata_min_layer_keep_ratio", mlkr == args.expect_min_layer_keep_ratio,
                           f"metadata={mlkr!r}"))
            # The floor is what makes per-tensor keep rates non-uniform; if the smallest tensor
            # rate sits below it, the floor was not applied.
            ok_floor = (s["min_tensor_keep_rate"] is not None
                        and s["min_tensor_keep_rate"] >= args.expect_min_layer_keep_ratio - 1e-9)
            checks.append(("per_tensor_floor_respected", ok_floor,
                           f"min tensor keep rate={s['min_tensor_keep_rate']:.6f}"))

        checks.append(("no_empty_tensors", s["n_empty_tensors"] == 0,
                       f"{s['n_empty_tensors']} all-zero tensors"))

        pooling = s["metadata"].get("pooling_mode")
        checks.append(("pooling_mode_recorded", pooling is not None, f"pooling_mode={pooling!r}"))

        j = jaccard(masks, ref_masks) if set(masks) == set(ref_masks) else None

        print(f"  tensors={s['n_tensors']}  elements={s['n_elements']:,}  "
              f"keep={s['keep_rate']:.6f}  k={s['k_kept']:,}")
        print(f"  per-tensor keep rate: min={s['min_tensor_keep_rate']:.6f} "
              f"max={s['max_tensor_keep_rate']:.6f}")
        print(f"  metadata: pooling_mode={pooling!r} min_layer_keep_ratio={mlkr!r} "
              f"sparsity={s['metadata'].get('sparsity_percent', s['metadata'].get('sparsity'))!r} "
              f"target_step={s['metadata'].get('target_step')!r}")
        if j is not None:
            print(f"  Jaccard vs reference: {j:.6f}")
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            if not ok:
                failures.append((path, name, detail))

        rec = {k: v for k, v in s.items() if k != "_per"}
        rec["jaccard_vs_reference"] = j
        rec["checks"] = [{"name": n, "ok": bool(o), "detail": d} for n, o, d in checks]
        out["masks"].append(rec)
        del masks

    out["failures"] = [{"path": a, "check": b, "detail": c} for a, b, c in failures]
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out_json}")

    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED — do not train on these masks.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

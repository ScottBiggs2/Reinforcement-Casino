#!/usr/bin/env python3
"""Inspect sparse mask candidate files without loading a model."""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import torch


def mask_dict(obj: Any):
    if isinstance(obj, dict) and isinstance(obj.get("masks"), dict):
        return "masks", obj["masks"]
    if isinstance(obj, dict) and isinstance(obj.get("mask"), dict):
        return "mask", obj["mask"]
    if isinstance(obj, dict):
        tensors = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if tensors:
            return "raw_tensors", tensors
    return "unknown", {}


def main() -> int:
    for path in sys.argv[1:]:
        print(f"\n== {path} ==")
        print(f"exists={os.path.exists(path)} size={os.path.getsize(path) if os.path.exists(path) else None}")
        if not os.path.exists(path):
            continue

        obj = torch.load(path, map_location="cpu")
        print(f"top_type={type(obj).__name__}")
        if isinstance(obj, dict):
            print(f"top_keys={list(obj.keys())[:30]} nkeys={len(obj)}")
            for key in ("metadata", "meta", "config", "args", "sparsity", "method", "target_step"):
                if key in obj and not torch.is_tensor(obj[key]):
                    print(f"{key}={obj[key]}")

        container, masks = mask_dict(obj)
        print(f"container={container} ntensors={len(masks)}")

        total = 0
        nonzero = 0
        binary = True
        min_value = None
        max_value = None
        samples = []
        layer_counts: dict[int, list[int]] = {}

        for name, value in masks.items():
            if not torch.is_tensor(value):
                continue
            tensor = value.detach().cpu()
            n = tensor.numel()
            nz = int((tensor != 0).sum().item())
            total += n
            nonzero += nz

            flat = tensor.flatten()
            if n:
                f = flat.float()
                lo = float(f.min())
                hi = float(f.max())
                min_value = lo if min_value is None else min(min_value, lo)
                max_value = hi if max_value is None else max(max_value, hi)

            if not bool(((tensor == 0) | (tensor == 1)).all()):
                binary = False

            match = re.search(r"layers\.(\d+)", name)
            if match:
                layer = int(match.group(1))
                layer_counts.setdefault(layer, [0, 0])
                layer_counts[layer][0] += nz
                layer_counts[layer][1] += n

            if len(samples) < 6:
                uniques = torch.unique(flat[:10000])[:10].tolist()
                samples.append(
                    (
                        name,
                        tuple(tensor.shape),
                        str(tensor.dtype),
                        round(nz / n, 8) if n else None,
                        [float(x) for x in uniques],
                    )
                )

        print(
            "overall_nonzero_frac="
            f"{round(nonzero / total, 8) if total else None} "
            f"zero_frac={round(1 - nonzero / total, 8) if total else None} total={total}"
        )
        print(f"binary_01={binary} min={min_value} max={max_value}")
        if layer_counts:
            first = [(i, round(layer_counts[i][0] / layer_counts[i][1], 8)) for i in sorted(layer_counts)[:4]]
            last = [(i, round(layer_counts[i][0] / layer_counts[i][1], 8)) for i in sorted(layer_counts)[-4:]]
            print(f"layers={len(layer_counts)} first={first} last={last}")
        for sample in samples:
            print(f"sample={sample}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

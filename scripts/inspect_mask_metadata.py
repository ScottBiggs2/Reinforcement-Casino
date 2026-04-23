#!/usr/bin/env python3
"""Print mask file metadata and key counts, avoiding tensor scans."""

from __future__ import annotations

import json
import os
import sys

import torch


for path in sys.argv[1:]:
    print(f"\n== {path} ==")
    print(f"exists={os.path.exists(path)} size={os.path.getsize(path) if os.path.exists(path) else None}")
    if not os.path.exists(path):
        continue
    try:
        obj = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    print(f"top_type={type(obj).__name__}")
    if not isinstance(obj, dict):
        continue
    print(f"top_keys={list(obj.keys())[:30]} nkeys={len(obj)}")
    metadata = obj.get("metadata") or obj.get("meta") or {}
    print("metadata=" + json.dumps(metadata, indent=2, sort_keys=True)[:5000])
    masks = obj.get("masks") or obj.get("mask") or {
        k: v for k, v in obj.items() if torch.is_tensor(v)
    }
    print(f"mask_count={len(masks)}")
    print(f"first_mask_keys={list(masks.keys())[:10]}")

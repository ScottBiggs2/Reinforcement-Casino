#!/usr/bin/env python
"""Fail fast if a base model cannot be resolved from the offline HF cache.

Why this exists: on 2026-07-30 job 235953 (Qwen3-32B p1) queued, started, loaded
the dataset, and only then died in AutoTokenizer.from_pretrained because
Qwen/Qwen3-32B was never actually in the AICR cache -- the `hf download` that was
supposed to fetch it hangs with zero bytes written. The afterok chain read that
as a hard crash and cancelled the six downstream jobs. A config-only check is not
enough either: the small files can be present while a 3.9 GB shard is missing or
truncated, which fails ~30 min into training instead of at second 5.

Checks, in order: config resolves, tokenizer materialises, the weight index
resolves, and every shard the index names is present with a non-zero size.

Usage: python scripts/preflight_base_model.py Qwen/Qwen3-32B
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: preflight_base_model.py <model_id_or_path>", file=sys.stderr)
        return 2
    model_id = sys.argv[1]
    offline = os.environ.get("HF_HUB_OFFLINE", "0")
    print(f"[preflight] model={model_id} HF_HUB_OFFLINE={offline} HF_HOME={os.environ.get('HF_HOME')}")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.utils import cached_file

    cfg = AutoConfig.from_pretrained(model_id)
    print(f"[preflight] config OK: {cfg.model_type}, {getattr(cfg, 'num_hidden_layers', '?')} layers")

    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"[preflight] tokenizer OK: vocab {len(tok)}")

    # Sharded checkpoints name their shards in an index; a single-file checkpoint
    # has no index, so fall back to the lone weights file.
    try:
        index_path = cached_file(model_id, "model.safetensors.index.json")
    except Exception:
        index_path = None

    if index_path:
        with open(index_path) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
        missing = []
        for shard in shards:
            try:
                path = cached_file(model_id, shard)
                if os.path.getsize(path) == 0:
                    missing.append(f"{shard} (zero bytes)")
            except Exception as exc:  # noqa: BLE001 - report whatever the hub raised
                missing.append(f"{shard} ({type(exc).__name__})")
        if missing:
            print(f"[preflight] FATAL: {len(missing)}/{len(shards)} shards unusable:", file=sys.stderr)
            for entry in missing:
                print(f"  - {entry}", file=sys.stderr)
            return 1
        print(f"[preflight] weights OK: {len(shards)} shards all present")
    else:
        path = cached_file(model_id, "model.safetensors")
        print(f"[preflight] weights OK: single file, {os.path.getsize(path)} bytes")

    print("[preflight] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

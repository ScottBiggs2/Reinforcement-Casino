#!/usr/bin/env python3
"""Patch chat_template into already-saved masked checkpoints.

`AutoTokenizer.save_pretrained()` on some transformers versions drops
`chat_template`; evaluators then silently disable chat formatting and
Instruct models score near zero on everything that needs instructions.

This script reads chat_template from the source HF tokenizer and writes it
into each masked checkpoint's tokenizer_config.json in place.

Usage:
    python src/evaluation/patch_masked_tokenizers.py \\
        --source meta-llama/Llama-3.1-8B-Instruct \\
        --dirs /scratch/$USER/masked_ckpts/*
"""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dirs", nargs="+", required=True,
                   help="Masked checkpoint directories to patch.")
    return p.parse_args()


def main():
    args = parse_args()
    src = AutoTokenizer.from_pretrained(args.source)
    template = getattr(src, "chat_template", None)
    if not template:
        raise RuntimeError(f"Source tokenizer {args.source} has no chat_template.")
    print(f"[patch] source chat_template len={len(template)}")

    for d in args.dirs:
        cfg_path = Path(d) / "tokenizer_config.json"
        if not cfg_path.exists():
            print(f"[skip] {d}: no tokenizer_config.json")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        had = "chat_template" in cfg
        cfg["chat_template"] = template
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[patch] {d}  (was_present={had})")


if __name__ == "__main__":
    main()

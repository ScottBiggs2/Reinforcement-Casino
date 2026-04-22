#!/usr/bin/env python3
"""Apply a binary MLP mask to a base HF model and save the masked checkpoint.

The saved directory is a standard HF checkpoint (config + safetensors + tokenizer)
that drop-in replaces the base `--model` for any evaluator that takes a model path,
including the vLLM backend in `run_all_benchmarks.py`.

Usage:
    python src/evaluation/apply_mask_and_save.py \\
        --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --mask /scratch/$USER/.../cold_cav_dpo.pt \\
        --output_dir /scratch/$USER/masked_ckpts/Cold-CAV-DPO
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.analysis.probe_analysis import load_mask  # reuses the wrapper/raw-dict logic


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"[save] base={args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    print(f"[save] mask={args.mask}")
    mask = load_mask(args.mask)

    applied = 0
    unmatched = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            m = mask[name]
            if not torch.all((m == 0) | (m == 1)):
                raise ValueError(f"Non-binary mask for {name}")
            param.data.mul_(m.to(param.device, dtype=param.dtype))
            applied += 1
        for name in mask:
            if not any(p_name == name for p_name, _ in model.named_parameters()):
                unmatched.append(name)
    print(f"[save] applied mask to {applied} tensors; "
          f"{len(unmatched)} mask keys had no match")

    print(f"[save] writing to {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tok.save_pretrained(args.output_dir)

    # Some transformers versions drop chat_template during save_pretrained.
    # Force-write it into tokenizer_config.json so instruct-model evals work.
    template = getattr(tok, "chat_template", None)
    if template:
        import json
        cfg_path = os.path.join(args.output_dir, "tokenizer_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("chat_template") != template:
            cfg["chat_template"] = template
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[save] patched chat_template into {cfg_path}")
    print("[save] done")


if __name__ == "__main__":
    main()

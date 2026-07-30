#!/usr/bin/env python3
"""Materialize a grafted checkpoint: base + alpha * (Delta-theta restricted to a mask).

Grafting (Panigrahi et al., ICML 2023): transplant the fine-tuning update on a
sparse coordinate set M onto the untouched base model,

    delta_only:      theta_out = theta_base + alpha * (theta_ft - theta_base) * M
    anti_delta_only: theta_out = theta_base + alpha * (theta_ft - theta_base) * (1 - M)

Unlike apply_mask_and_save.py (zero-out), this never destroys base weights:
parameters outside the patched set keep their base values, parameters absent
from the mask dict are left at base too. The saved directory is a standard HF
checkpoint usable by preference_eval.py / run_all_benchmarks.py.

Usage:
    python src/evaluation/apply_delta_patch_and_save.py \\
        --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --finetuned_model /scratch/$USER/.../checkpoint-500 \\
        --mask /scratch/$USER/.../oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt \\
        --patch_mode delta_only --alpha 1.0 \\
        --output_dir /scratch/$USER/graft_ckpts/graft_oracle_a1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.analysis.probe_analysis import load_mask  # wrapper/raw-dict logic


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--finetuned_model", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--patch_mode", choices=["delta_only", "anti_delta_only"],
                   default="delta_only")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Scale on the patched delta (task-arithmetic dose). "
                        "1.0 = full graft; 0.5 = half; -1.0 = negation.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print(f"[graft] base = {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    print(f"[graft] finetuned = {args.finetuned_model}")
    ft = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    ft_sd = ft.state_dict()

    print(f"[graft] mask = {args.mask}")
    mask = load_mask(args.mask)

    patched = 0
    skipped_shape = 0
    delta_norm_sq = 0.0
    patched_norm_sq = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in ft_sd:
                continue
            ft_w = ft_sd[name]
            if ft_w.shape != param.shape:
                skipped_shape += 1
                continue
            # fp32 delta to avoid bf16 cancellation on tiny updates
            delta = ft_w.to(torch.float32) - param.data.to(torch.float32)
            delta_norm_sq += float((delta * delta).sum())
            if name not in mask:
                # No mask entry -> stays at base under BOTH modes. This matches
                # apply_mask's skip semantics and keeps the operator honest:
                # only coordinates the mask speaks about are ever patched.
                continue
            m = mask[name]
            if m.shape != param.shape:
                skipped_shape += 1
                continue
            m = m.to(torch.float32)
            if not torch.all((m == 0) | (m == 1)):
                raise ValueError(f"Non-binary mask for {name}")
            if args.patch_mode == "anti_delta_only":
                m = 1.0 - m
            sel = delta * m
            patched_norm_sq += float((sel * sel).sum())
            param.data.add_((args.alpha * sel).to(param.dtype))
            patched += 1

    frac = patched_norm_sq / delta_norm_sq if delta_norm_sq > 0 else float("nan")
    print(f"[graft] mode={args.patch_mode} alpha={args.alpha}")
    print(f"[graft] patched {patched} tensors; {skipped_shape} shape-skipped")
    print(f"[graft] ||delta on patched set||^2 / ||delta||^2 = {frac:.6f}")

    del ft, ft_sd
    print(f"[graft] writing {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tok.save_pretrained(args.output_dir)

    # keep chat_template (some transformers versions drop it on save)
    template = getattr(tok, "chat_template", None)
    if template:
        cfg_path = os.path.join(args.output_dir, "tokenizer_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("chat_template") != template:
            cfg["chat_template"] = template
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

    meta = {
        "base_model": args.base_model,
        "finetuned_model": args.finetuned_model,
        "mask": args.mask,
        "patch_mode": args.patch_mode,
        "alpha": args.alpha,
        "patched_tensors": patched,
        "delta_mass_fraction_on_patched_set": frac,
    }
    with open(os.path.join(args.output_dir, "graft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[graft] done")


if __name__ == "__main__":
    main()

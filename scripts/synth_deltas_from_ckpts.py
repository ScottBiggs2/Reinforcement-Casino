"""
Reconstruct `deltas_step_<N>.pt` files from saved HF checkpoints + base_state.pt.

Used when a dense DPO run was launched without delta logging through the desired
target step (e.g. dense_dpo_lightr1_llama8b only saved step-50 deltas, but we
have HF checkpoints at every multiple of 25 steps).

Mirrors `FlexibleCheckpointCallback.on_step_end` in src/full_training/DPO_train.py:
iterates `named_parameters()`, casts to float32 + cpu, subtracts base_state, saves.

Usage:
    python scripts/synth_deltas_from_ckpts.py \
        --base_state /scratch/.../deltas/.../base_state.pt \
        --ckpt_root  /scratch/.../checkpoints/.../ \
        --steps 100 150 200 \
        --out_dir    /scratch/.../deltas/.../
"""
import argparse
import gc
import os
import sys

import torch
from transformers import AutoModelForCausalLM


def synth_one(base_state_cpu_fp32, ckpt_dir, out_path):
    print(f"  Loading HF checkpoint: {ckpt_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.bfloat16,  # match training dtype, then upcast in subtraction
        device_map=None,
        low_cpu_mem_usage=True,
    )

    deltas = {}
    missing_in_base = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            current = param.detach().float().cpu()
            if name not in base_state_cpu_fp32:
                missing_in_base.append(name)
                continue
            deltas[name] = current - base_state_cpu_fp32[name]

    if missing_in_base:
        print(f"  WARN: {len(missing_in_base)} params in ckpt missing in base_state (skipped). First 3: {missing_in_base[:3]}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"  Saving {len(deltas)} delta tensors -> {out_path}")
    torch.save(deltas, out_path)

    # free memory
    del model, deltas
    gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_state", required=True, help="Path to base_state.pt (W_0 fp32 cpu dict)")
    p.add_argument("--ckpt_root", required=True, help="Dir containing checkpoint-<N> subdirs")
    p.add_argument("--steps", required=True, type=int, nargs="+", help="Target steps to synthesize (e.g. 100 150 200)")
    p.add_argument("--out_dir", required=True, help="Where to save deltas_step_<N>.pt")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not os.path.isfile(args.base_state):
        print(f"FATAL: base_state not found: {args.base_state}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading base_state: {args.base_state}")
    base = torch.load(args.base_state, map_location="cpu")
    print(f"  -> {len(base)} tensors")

    for step in args.steps:
        ckpt_dir = os.path.join(args.ckpt_root, f"checkpoint-{step}")
        out_path = os.path.join(args.out_dir, f"deltas_step_{step}.pt")
        if not os.path.isdir(ckpt_dir):
            print(f"SKIP step {step}: ckpt missing at {ckpt_dir}")
            continue
        if os.path.isfile(out_path) and not args.overwrite:
            print(f"SKIP step {step}: {out_path} already exists (use --overwrite to force)")
            continue
        print(f"\n[step {step}]")
        synth_one(base, ckpt_dir, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

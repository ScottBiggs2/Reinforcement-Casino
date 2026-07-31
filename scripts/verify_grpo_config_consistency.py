#!/usr/bin/env python3
"""
Equal-footing gate for a training cell (GRPO *or* DPO): refuse to analyze unless every arm trained
under the same schedule and hyperparameters. Objective-neutral — DPOConfig lacks the GRPO-only
fields (num_generations, generation_batch_size, max_completion_length), which then compare as None
uniformly across arms and don't trip the gate.

Why this exists: the dense arm runs through ``GRPO_train.py`` and the sparse arms through
``sparse_grpo_bsr.py``; the two have *divergent defaults* (grad_accum 4 vs 8, max_grad_norm 0.1 vs
1.0, ``warmup_ratio`` vs ``warmup_steps`` — a different axis — batch 2 vs 1). A per-step trajectory
comparison d(M) is only meaningful if dense and sparse are on identical footing. This is the GRPO
analogue of the E2 DPO config guard (RESULTS.md §2), which caught real drift, and of
``verify_masks_full_coverage.py`` for coverage.

Authoritative source is ``training_args.bin`` (the pickled ``GRPOConfig``), NOT ``run_manifest.json``
— the manifest carries ~12 fields and lacks warmup/seed/clip. Warmup is compared as *effective
steps* (``warmup_steps`` if > 0 else ``round(warmup_ratio * max_steps)``) so the ratio-vs-steps axis
mismatch is caught rather than hidden.

Clipping caveat: ``max_grad_norm`` here is the HF Trainer's *global* clip. Sparse arms additionally
run ``SparseAdamW``'s *per-parameter* clip, which is not in ``training_args.bin``. The guard reports
each arm's optimizer (from ``run_manifest.json``) and FAILS loudly if the clipping regime is not
uniform, because that is a footing decision, not a default.

Usage:
  python scripts/verify_grpo_config_consistency.py \
      --arm dense=/scratch/$USER/.../grpo_llama31_openr1_dense \
      --arm oracle=/scratch/$USER/.../grpo_llama31_openr1_oracle \
      --arm random=/scratch/$USER/.../grpo_llama31_openr1_random \
      --arm mag50=... --arm mag100=... --arm mag250=... \
      --out_json /scratch/$USER/rl_casino_analysis/grpo_footing_gate.json

Exits non-zero if any arm disagrees with the reference (first arm, or --reference).
"""

import argparse
import glob
import json
import os
import sys

import torch

# Fields that MUST match across arms for d(M) to be a fair comparison. optim/output_dir/run_name
# are intentionally excluded (sparse arms log adamw_torch_fused but use SparseAdamW; paths differ).
FOOTING_FIELDS = [
    "max_steps",
    "learning_rate",
    "lr_scheduler_type",
    "max_grad_norm",
    "gradient_accumulation_steps",
    "per_device_train_batch_size",
    "beta",
    "num_generations",
    "generation_batch_size",
    "max_prompt_length",
    "max_completion_length",
    "seed",
    "data_seed",
    "bf16",
    "fp16",
    # DPO reference term: a static frozen-ref deep-copy either way, but precompute vs live can
    # differ numerically and the author flags it as "a different objective". Absent on GRPOConfig
    # (compares as None uniformly). Must be identical across DPO arms.
    "precompute_ref_log_probs",
]


def _find_training_args(arm_dir):
    """Locate training_args.bin under an arm's output dir (root, checkpoints/, latest checkpoint-*)."""
    cands = [
        os.path.join(arm_dir, "training_args.bin"),
        os.path.join(arm_dir, "checkpoints", "training_args.bin"),
    ]
    ckpts = sorted(
        glob.glob(os.path.join(arm_dir, "checkpoints", "checkpoint-*", "training_args.bin")),
        key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]),
    )
    if ckpts:
        cands.append(ckpts[-1])
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _effective_warmup_steps(ta):
    ws = int(getattr(ta, "warmup_steps", 0) or 0)
    if ws > 0:
        return ws
    wr = float(getattr(ta, "warmup_ratio", 0.0) or 0.0)
    ms = int(getattr(ta, "max_steps", 0) or 0)
    return round(wr * ms)


def _read_arm(label, arm_dir):
    ta_path = _find_training_args(arm_dir)
    if ta_path is None:
        raise SystemExit(f"[{label}] no training_args.bin under {arm_dir}")
    ta = torch.load(ta_path, weights_only=False, map_location="cpu")

    cfg = {f: getattr(ta, f, None) for f in FOOTING_FIELDS}
    cfg["effective_warmup_steps"] = _effective_warmup_steps(ta)
    # transparency (not compared directly; effective_warmup_steps is)
    cfg["_warmup_ratio_raw"] = getattr(ta, "warmup_ratio", None)
    cfg["_warmup_steps_raw"] = getattr(ta, "warmup_steps", None)
    acc = getattr(ta, "accelerator_config", None)
    cfg["use_seedable_sampler"] = getattr(acc, "use_seedable_sampler", None) if acc else None

    # optimizer / clipping regime from the run manifest, if present. The per-param clip level is
    # what matters, not the optimizer name: under the single-clip regime SparseAdamW's per-param
    # clip is 0 (off), so a sparse arm is on equal footing with dense (which has no per-param clip).
    manifest = {}
    for man in (os.path.join(arm_dir, "run_manifest.json"),
                os.path.join(arm_dir, "checkpoints", "run_manifest.json")):
        if os.path.isfile(man):
            with open(man) as f:
                manifest = json.load(f) or {}
            break
    cfg["_optimizer"] = manifest.get("optimizer")
    # Missing field (e.g. a dense arm) ⇒ no per-param clip ⇒ 0.0.
    cfg["_perparam_clip"] = float(manifest.get("sparse_adamw_max_grad_norm", 0.0) or 0.0)
    cfg["_training_args_path"] = ta_path
    return cfg


COMPARED = FOOTING_FIELDS + ["effective_warmup_steps", "use_seedable_sampler"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True,
                   help="label=output_dir, repeatable (dense, oracle, random, mag50/100/250).")
    p.add_argument("--reference", default=None, help="Label to compare against (default: first arm).")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    arms = {}
    order = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects label=dir, got {spec!r}")
        label, d = spec.split("=", 1)
        if not os.path.isdir(d):
            raise SystemExit(f"[{label}] not a directory: {d}")
        arms[label] = _read_arm(label, d)
        order.append(label)

    ref_label = args.reference or order[0]
    if ref_label not in arms:
        raise SystemExit(f"--reference {ref_label!r} not among arms {order}")
    ref = arms[ref_label]

    all_ok = True
    mismatches = {}
    for label in order:
        if label == ref_label:
            continue
        diffs = {k: {"ref": ref.get(k), label: arms[label].get(k)}
                 for k in COMPARED if arms[label].get(k) != ref.get(k)}
        if diffs:
            all_ok = False
            mismatches[label] = diffs

    # Clipping-regime uniformity: the global clip (max_grad_norm) is already in COMPARED; here we
    # require the per-param clip LEVEL to match across arms too. Under the single-clip regime that
    # is 0.0 everywhere (dense has none; sparse disables it). Any arm with a live per-param clip
    # while others don't means dense and sparse are clipped differently.
    clip_levels = {l: arms[l]["_perparam_clip"] for l in order}
    clip_uniform = len(set(clip_levels.values())) == 1
    if not clip_uniform:
        all_ok = False

    # ---- report ----
    print(f"GRPO equal-footing gate — reference: {ref_label}\n")
    hdr = ["field"] + order
    print("  ".join(f"{h:<22}" for h in hdr))
    print("-" * (24 * len(hdr)))
    for k in COMPARED:
        row = [k] + [str(arms[l].get(k)) for l in order]
        flag = "" if all(arms[l].get(k) == ref.get(k) for l in order) else "  ✗"
        print("  ".join(f"{c:<22}" for c in row) + flag)
    print("\noptimizer / per-param clip level (0.0 = single-clip regime):")
    for l in order:
        print(f"  {l:<10} optimizer={arms[l]['_optimizer']}  "
              f"sparseadamw_perparam_clip={arms[l]['_perparam_clip']}")
    if not clip_uniform:
        print("  ✗ clipping regime is NOT uniform — some arm applies a SparseAdamW per-param clip "
              "that others don't. Set --sparse_adamw_max_grad_norm=0 so the Trainer global clip is "
              "the only clip in every arm.")

    print()
    if all_ok:
        print(f"✓ ALL {len(order)} ARMS ON EQUAL FOOTING")
    else:
        print("✗ FOOTING MISMATCH")
        for label, diffs in mismatches.items():
            for k, v in diffs.items():
                print(f"    {label}.{k}: {v[label]!r}  vs ref {v['ref']!r}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"reference": ref_label, "all_ok": all_ok,
                       "clip_uniform": clip_uniform, "arms": arms,
                       "mismatches": mismatches}, f, indent=2, default=str)
        print(f"wrote {args.out_json}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

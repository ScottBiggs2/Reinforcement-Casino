"""Multi-seed jittered mask generator.

For variance studies of the otherwise-deterministic Oracle and Magnitude
masks. The expensive parts (loading checkpoints / delta files) happen ONCE;
multiple jittered seed variants are then derived from the same accumulated
scores.

Usage examples:
  # Oracle: load 2 ckpts once, jitter 3 times.
  python -m src.warm_start.multi_seed_mask_gen oracle \
      --initial_model meta-llama/Llama-3.1-8B-Instruct \
      --final_model /scratch/.../checkpoint-500 \
      --sparsity_percent 97.5 \
      --min_layer_keep_ratio 0.0025 \
      --seeds 42 43 44 \
      --jitter_rel 1e-3 \
      --output_pattern '/scratch/.../oracle_dpo_tulu3_step500_sp97.5_seed{seed}.pt'

  # Magnitude: walk delta files in chronological order, snapshot at each
  # requested target_step, jitter N times per snapshot.
  python -m src.warm_start.multi_seed_mask_gen magnitude \
      --delta_log_dir /scratch/.../deltas/... \
      --target_steps 50 100 200 \
      --sparsity_percent 97.5 \
      --min_layer_keep_ratio 0.0025 \
      --seeds 42 43 44 \
      --jitter_rel 1e-3 \
      --output_pattern '/scratch/.../warm_magnitude_dpo_tulu3_step{step}_sp97.5_seed{seed}.pt'
"""
import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    pooling_metadata,
    save_masks,
)
from src.warm_start.checkpoint_diff_mask_finder import load_state_dict
from src.warm_start.even_better_mask_finder import (
    choose_score_device,
    load_deltas_streaming,
)


def _jitter_and_save(scores, seeds, jitter_rel, sparsity, score_device,
                     min_layer_keep_ratio, output_pattern, base_meta,
                     extra_meta_per_seed=None):
    """For each seed: jitter scores in-place-on-copy, topk, save mask, skip if file exists."""
    stds = {n: float(t.std().item()) for n, t in scores.items()}

    for seed in seeds:
        out = output_pattern.format(seed=seed, **(extra_meta_per_seed or {}))
        if os.path.exists(out):
            print(f"[skip] {out} already exists")
            continue

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        jittered = {}
        for name, sc in scores.items():
            eps = stds[name] * jitter_rel
            noise = torch.randn(sc.shape, generator=gen, dtype=sc.dtype).to(sc.device)
            jittered[name] = sc + noise * eps

        masks = create_mask_from_scores_gpu_efficient(
            jittered, sparsity, score_device,
            local_pool=False, min_layer_keep_ratio=min_layer_keep_ratio,
        )
        for n in masks:
            masks[n] = masks[n].to(torch.bool)

        meta = dict(base_meta)
        meta.update({"seed": seed, "jitter_rel": jitter_rel,
                     **pooling_metadata(local_pool=False,
                                        min_layer_keep_ratio=min_layer_keep_ratio)})
        save_masks(masks, out, meta)
        del jittered, masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_oracle(args):
    print(f"\n=== Multi-seed Oracle (n={len(args.seeds)} seeds) ===")
    print(f"Initial:   {args.initial_model}")
    print(f"Final:     {args.final_model}")
    print(f"Seeds:     {args.seeds}")

    initial_sd = load_state_dict(args.initial_model, device="cpu")
    final_sd = load_state_dict(args.final_model, device="cpu")

    print("Computing |Δθ| once...")
    scores = {}
    for name in final_sd:
        if name in initial_sd:
            diff = (final_sd[name].to(torch.float32)
                    - initial_sd[name].to(torch.float32)).abs()
            scores[name] = diff
    print(f"  {len(scores)} parameter tensors")

    del initial_sd, final_sd
    gc.collect()

    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    # Move scores to device for fast topk later.
    if device == "cuda":
        scores = {n: t.to(device) for n, t in scores.items()}

    base_meta = {
        "method": "checkpoint_difference_ground_truth",
        "sparsity_percent": args.sparsity_percent,
        "initial_model": args.initial_model,
        "final_model": args.final_model,
        "mlp_only": False,
        "device": device,
    }
    _jitter_and_save(scores, args.seeds, args.jitter_rel,
                     args.sparsity_percent, device, args.min_layer_keep_ratio,
                     args.output_pattern, base_meta)


def run_magnitude(args):
    print(f"\n=== Multi-seed Magnitude (target_steps={args.target_steps}, "
          f"n={len(args.seeds)} seeds) ===")

    if os.environ.get("RL_CASINO_WARM_MASK_SCORE_DEVICE") == "cpu":
        args.force_cpu = True
    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    score_device = choose_score_device(device)
    print(f"score_device={score_device}")

    # Walk ALL deltas up to max(target_steps). Accumulator runs in order; at
    # each step in target_steps, snapshot+jitter+save N masks.
    max_target = max(args.target_steps)
    steps_and_paths = load_deltas_streaming(args.delta_log_dir, max_target)
    target_set = set(args.target_steps)
    base_meta_template = {
        "method": "absolute_magnitude_streaming",
        "sparsity_percent": args.sparsity_percent,
        "mlp_only": False,
        "device": device,
    }

    aggregated = {}
    for step_idx, (step, dp) in enumerate(steps_and_paths):
        print(f"  [{step_idx + 1}/{len(steps_and_paths)}] step={step}  path={dp}")
        d = torch.load(dp, map_location=score_device)
        if not aggregated:
            for n in d:
                aggregated[n] = torch.zeros_like(d[n], device=score_device)
        for n in aggregated:
            if n in d:
                aggregated[n] += d[n].abs()
        del d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if step in target_set:
            print(f"  → snapshot at step={step}; jittering for seeds={args.seeds}")
            base_meta = dict(base_meta_template)
            base_meta["target_step"] = step
            _jitter_and_save(
                aggregated, args.seeds, args.jitter_rel,
                args.sparsity_percent, score_device, args.min_layer_keep_ratio,
                args.output_pattern, base_meta,
                extra_meta_per_seed={"step": step},
            )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    common = lambda p: (
        p.add_argument("--sparsity_percent", type=float, default=97.5),
        p.add_argument("--min_layer_keep_ratio", type=float,
                       default=DEFAULT_MIN_LAYER_KEEP_RATIO),
        p.add_argument("--seeds", type=int, nargs="+", required=True),
        p.add_argument("--jitter_rel", type=float, default=1e-3),
        p.add_argument("--output_pattern", required=True,
                       help="Format string with {seed}; magnitude also gets {step}"),
        p.add_argument("--force_cpu", action="store_true"),
    )

    po = sub.add_parser("oracle")
    po.add_argument("--initial_model", required=True)
    po.add_argument("--final_model", required=True)
    common(po)

    pm = sub.add_parser("magnitude")
    pm.add_argument("--delta_log_dir", required=True)
    pm.add_argument("--target_steps", type=int, nargs="+", required=True)
    common(pm)

    args = ap.parse_args()
    if args.mode == "oracle":
        run_oracle(args)
    elif args.mode == "magnitude":
        run_magnitude(args)


if __name__ == "__main__":
    main()

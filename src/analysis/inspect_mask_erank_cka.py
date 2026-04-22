#!/usr/bin/env python3
"""Bundled per-layer erank + CKA inspector for a list of masks.

One model load. One dense forward pass. Then for each mask:
  - weighted erank per matched layer  (SVD on W * mask)
  - CKA vs dense activations per layer (collect masked activations, then
    compute linear CKA against the already-cached dense activations)
  - restore the model and move to the next mask

Output: one JSON with `per_mask: {label: {erank: {layer: v}, cka: {...}}}`,
plus a summary heatmap PNG.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from cold_start.utils.activation_hooks import FeatureExtractor, infer_model_input_device
from cold_start.mask_to_cka import (
    apply_mask as cka_apply_mask,
    collect_activations,
    compute_layerwise_cka,
    load_calibration_samples,
    load_masks,
    restore_weights,
    set_seed,
    DEFAULT_CALIBRATION_DATASET,
)
from analysis.probe_analysis import _layer_index


def erank_from_singular_values(sigma: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank = exp(H(p)), with p_i = sigma_i / sum_j sigma_j.

    Matches Scott's convention in src/cold_start/export_layer_metrics_csv.py
    (probabilities from raw singular values, not squared).
    """
    s = sigma.float()
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum().item()
    return float(math.exp(entropy))


def weighted_erank_for_mask(mask_dict: dict, base_params: dict,
                            device: torch.device,
                            layer_substr: str) -> dict:
    """erank of (base_weight * mask) for every layer whose name contains `layer_substr`."""
    per_layer = {}
    for name, m in mask_dict.items():
        if layer_substr not in name or name not in base_params:
            continue
        w = base_params[name]
        if w.shape != m.shape or m.dim() != 2:
            continue
        masked = (w.to(device=device, dtype=torch.float32)
                  * m.to(device=device, dtype=torch.float32))
        try:
            sigma = torch.linalg.svdvals(masked)
        except RuntimeError:
            sigma = torch.linalg.svdvals(masked.cpu())
        max_rank = int(min(w.shape))
        erank = erank_from_singular_values(sigma)
        per_layer[name] = {
            "erank": erank,
            "erank_norm": erank / max_rank if max_rank > 0 else 0.0,
            "kept_frac": float(m.float().mean().item()),
            "rank_upper_bound": max_rank,
        }
    return per_layer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--masks_json", required=True,
                   help="List of {label, path} specifying the masks to inspect.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layer_substr", default="down_proj",
                   help="Only compute metrics for layers whose name contains "
                        "this substring (default: down_proj, matches probe hook)")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_name", default=DEFAULT_CALIBRATION_DATASET)
    p.add_argument("--erank_device", default="cuda",
                   help="Device for SVD (default: cuda if available)")
    p.add_argument("--skip_erank", action="store_true")
    p.add_argument("--skip_cka", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # --- mask list --------------------------------------------------------
    with open(args.masks_json) as f:
        specs = json.load(f)
    for spec in specs:
        if not os.path.exists(spec["path"]):
            raise FileNotFoundError(f"{spec['label']}: missing {spec['path']}")
    print(f"[main] inspecting {len(specs)} masks:")
    for s in specs:
        print(f"       {s['label']:20s} ← {s['path']}")

    # --- model ------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n[main] Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto",
    )
    model.config.use_cache = False
    model.eval()
    input_device = infer_model_input_device(model)
    erank_device = torch.device(args.erank_device if torch.cuda.is_available()
                                else "cpu")
    print(f"[main] model device={input_device}, erank device={erank_device}")

    # Snapshot base params once for weighted-erank.
    base_params = {n: p.detach() for n, p in model.named_parameters()}

    # --- CKA setup: collect dense activations once ------------------------
    results = {"args": vars(args), "per_mask": {}}
    dense_acts = None
    if not args.skip_cka:
        chosen_texts, _ = load_calibration_samples(
            args.n_samples, seed=args.seed, dataset_name=args.dataset_name,
        )
        extractor = FeatureExtractor()
        extractor.register(model)
        print(f"\n[main] Collecting dense activations "
              f"(n={args.n_samples}, max_len={args.max_length})")
        dense_acts = collect_activations(
            model, extractor, tok, chosen_texts, input_device,
            args.max_length, args.batch_size,
        )
        print(f"[main] dense activations: {len(dense_acts)} layers")
    else:
        extractor = None

    # --- per-mask loop ----------------------------------------------------
    for spec in specs:
        label = spec["label"]
        print(f"\n[main] ===== {label} =====", flush=True)
        masks, _meta = load_masks(spec["path"])
        per_mask = {}

        if not args.skip_erank:
            t0 = time.time()
            per_mask["erank"] = weighted_erank_for_mask(
                masks, base_params, erank_device, args.layer_substr,
            )
            n = len(per_mask["erank"])
            mean_er = np.mean([v["erank"] for v in per_mask["erank"].values()]) if n else float("nan")
            print(f"  [erank]   {n} layers, mean={mean_er:.2f} "
                  f"(took {time.time() - t0:.1f}s)")

        if not args.skip_cka:
            t0 = time.time()
            orig = cka_apply_mask(model, masks)
            print(f"  [cka]     collecting masked activations...")
            masked_acts = collect_activations(
                model, extractor, tok, chosen_texts, input_device,
                args.max_length, args.batch_size,
            )
            restore_weights(model, orig)
            cka_stats = compute_layerwise_cka(
                {k: v for k, v in dense_acts.items() if args.layer_substr in k},
                {k: v for k, v in masked_acts.items() if args.layer_substr in k},
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            per_mask["cka"] = cka_stats
            print(f"  [cka]     mean={cka_stats['mean_cka']:.4f} "
                  f"min={cka_stats['min_cka']:.4f} max={cka_stats['max_cka']:.4f} "
                  f"(took {time.time() - t0:.1f}s)")

        results["per_mask"][label] = per_mask

        # Incremental save so partial progress survives crashes.
        out_path = os.path.join(args.output_dir, "mask_erank_cka.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)

    if extractor is not None:
        extractor.remove()
    print(f"\n[main] JSON → {out_path}")

    # --- per-layer summary tables ----------------------------------------
    print_summary(results, layer_substr=args.layer_substr)


def print_summary(results: dict, layer_substr: str):
    per_mask = results.get("per_mask", {})
    if not per_mask:
        return

    all_idx = set()
    for blk in per_mask.values():
        for name in blk.get("erank", {}) | blk.get("cka", {}).get("per_layer", {}):
            idx = _layer_index(name)
            if idx >= 0:
                all_idx.add(idx)
    idxs = sorted(all_idx)
    if not idxs:
        return

    # erank table
    if any("erank" in b for b in per_mask.values()):
        print("\n=== erank per layer ===")
        header = f"{'mask':20s} " + " ".join(f"{i:>6d}" for i in idxs)
        print(header); print("-" * len(header))
        for label, blk in per_mask.items():
            er = blk.get("erank", {})
            idx_to_val = {_layer_index(n): v["erank"] for n, v in er.items()}
            cells = " ".join(f"{idx_to_val.get(i, float('nan')):6.1f}" for i in idxs)
            print(f"{label:20s} {cells}")

    # cka table
    if any("cka" in b for b in per_mask.values()):
        print("\n=== CKA(dense, masked) per layer ===")
        header = f"{'mask':20s} " + " ".join(f"{i:>6d}" for i in idxs)
        print(header); print("-" * len(header))
        for label, blk in per_mask.items():
            cka_pl = blk.get("cka", {}).get("per_layer", {})
            idx_to_val = {_layer_index(n): v for n, v in cka_pl.items() if v is not None}
            cells = " ".join(f"{idx_to_val.get(i, float('nan')):6.3f}" for i in idxs)
            print(f"{label:20s} {cells}")


if __name__ == "__main__":
    main()

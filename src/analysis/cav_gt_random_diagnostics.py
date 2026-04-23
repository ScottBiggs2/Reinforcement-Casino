#!/usr/bin/env python3
"""Diagnose Cold-CAV vs warm-magnitude oracle and a matched random mask.

Outputs:
  - structural per-layer effective rank for each mask
  - layer-wise activation CKA for CAV-vs-GT, random-vs-GT, CAV-vs-random
  - one combined CSV table for quick inspection
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))

from src.cold_start.mask_to_cka import (  # noqa: E402
    compute_layerwise_cka,
    collect_activations,
    load_calibration_samples,
    apply_mask,
    restore_weights,
    set_seed,
)
from src.cold_start.utils.activation_hooks import (  # noqa: E402
    FeatureExtractor,
    infer_model_input_device,
)
from src.utils.mask_utils import (  # noqa: E402
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    pooling_metadata,
    save_masks,
)


def load_mask(path: str) -> tuple[dict, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], data.get("metadata", {})
    if isinstance(data, dict):
        return data, {}
    raise ValueError(f"Unrecognized mask format: {path}")


def layer_index(name: str) -> int:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def canonical_layer_name(name: str) -> str:
    return name[:-7] if name.endswith(".weight") else name


def erank_from_singular_values(sigma: torch.Tensor) -> float:
    s2 = sigma.float().pow(2)
    total = s2.sum().clamp_min(1e-30)
    p = s2 / total
    nonzero = p > 0
    h = -(p[nonzero] * p[nonzero].log()).sum()
    return float(torch.exp(h).item())


def compute_erank(tensor: torch.Tensor, device: torch.device) -> float:
    x = tensor.to(device=device, dtype=torch.float32)
    try:
        sigma = torch.linalg.svdvals(x)
    except RuntimeError:
        sigma = torch.linalg.svdvals(x.cpu())
    return erank_from_singular_values(sigma)


def mask_erank_table(label: str, masks: dict, device: torch.device,
                     layer_pattern: str) -> dict:
    rows = {}
    for name, tensor in sorted(masks.items()):
        if layer_pattern not in name or tensor.dim() != 2:
            continue
        cname = canonical_layer_name(name)
        er = compute_erank(tensor, device)
        max_rank = min(tensor.shape)
        rows[cname] = {
            "label": label,
            "layer_index": layer_index(name),
            "erank": er,
            "erank_norm": er / float(max_rank) if max_rank else 0.0,
            "density": float(tensor.float().mean().item()),
            "shape": list(tensor.shape),
        }
        print(
            f"[erank] {label:12s} layer={rows[cname]['layer_index']:02d} "
            f"erank={er:.2f} norm={rows[cname]['erank_norm']:.4f} "
            f"density={rows[cname]['density']:.5f}",
            flush=True,
        )
    return rows


def generate_random_if_missing(args, gt_masks: dict):
    if os.path.exists(args.random_mask) and os.path.getsize(args.random_mask) > 0:
        print(f"[random] existing random mask: {args.random_mask}")
        return

    print(f"[random] generating matched random mask -> {args.random_mask}")
    torch.manual_seed(args.seed)
    scores = {
        name: torch.rand_like(mask.float())
        for name, mask in gt_masks.items()
    }
    random_masks = create_mask_from_scores_gpu_efficient(
        scores,
        args.sparsity_percent,
        device="cpu",
        local_pool=args.local_pool,
        min_layer_keep_ratio=args.min_layer_keep_ratio,
    )
    metadata = {
        "method": "random_baseline",
        "sparsity_percent": args.sparsity_percent,
        "seed": args.seed,
        "reference_mask": args.gt_mask,
        **pooling_metadata(
            local_pool=args.local_pool,
            min_layer_keep_ratio=args.min_layer_keep_ratio,
        ),
    }
    os.makedirs(os.path.dirname(args.random_mask) or ".", exist_ok=True)
    save_masks(random_masks, args.random_mask, metadata)


def collect_masked_acts(label: str, model, extractor, tokenizer, texts, masks,
                        input_device, args) -> dict:
    print(f"[cka] collecting activations for {label}", flush=True)
    original = apply_mask(model, masks)
    try:
        return collect_activations(
            model, extractor, tokenizer, texts, input_device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    finally:
        restore_weights(model, original)


def write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_cka(name: str, cka: dict):
    print(
        f"[cka] {name:14s} mean={cka['mean']:.4f} "
        f"min={cka['min']:.4f} max={cka['max']:.4f} "
        f"layers={cka['n_layers']} skipped={cka['n_skipped']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--cav_mask",
        default="/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt",
    )
    parser.add_argument(
        "--gt_mask",
        default="/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt",
    )
    parser.add_argument(
        "--random_mask",
        default="/scratch/xie.yiyi/rl_casino_masks/llama8b/random_baseline_dpo_sp97.5_seed42.pt",
    )
    parser.add_argument("--output_dir", default="/scratch/xie.yiyi/cav_gt_random_diagnostics")
    parser.add_argument("--sparsity_percent", type=float, default=97.5)
    parser.add_argument("--min_layer_keep_ratio", type=float, default=DEFAULT_MIN_LAYER_KEEP_RATIO)
    parser.add_argument("--local_pool", action="store_true")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer_pattern", default="down_proj")
    parser.add_argument("--erank_device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    for path in (args.cav_mask, args.gt_mask):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    print("[load] CAV:", args.cav_mask)
    cav_masks, cav_meta = load_mask(args.cav_mask)
    print("[load] GT :", args.gt_mask)
    gt_masks, gt_meta = load_mask(args.gt_mask)
    generate_random_if_missing(args, gt_masks)
    print("[load] random:", args.random_mask)
    random_masks, random_meta = load_mask(args.random_mask)

    erank_device = torch.device(
        args.erank_device if args.erank_device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    eranks = {
        "Cold-CAV-DPO": mask_erank_table("Cold-CAV-DPO", cav_masks, erank_device, args.layer_pattern),
        "Oracle-DPO": mask_erank_table("Oracle-DPO", gt_masks, erank_device, args.layer_pattern),
        "Random-DPO": mask_erank_table("Random-DPO", random_masks, erank_device, args.layer_pattern),
    }
    write_json(os.path.join(args.output_dir, "erank_by_mask.json"), eranks)

    print(f"[model] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.config.use_cache = False
    model.eval()
    input_device = infer_model_input_device(model)

    chosen_texts, _ = load_calibration_samples(args.n_samples, seed=args.seed)
    extractor = FeatureExtractor().register(model)
    try:
        acts_cav = collect_masked_acts("Cold-CAV-DPO", model, extractor, tokenizer,
                                       chosen_texts, cav_masks, input_device, args)
        acts_gt = collect_masked_acts("Oracle-DPO", model, extractor, tokenizer,
                                      chosen_texts, gt_masks, input_device, args)
        acts_random = collect_masked_acts("Random-DPO", model, extractor, tokenizer,
                                          chosen_texts, random_masks, input_device, args)
    finally:
        extractor.remove()

    comparisons = {
        "cav_vs_gt": (acts_cav, acts_gt, args.cav_mask, args.gt_mask),
        "random_vs_gt": (acts_random, acts_gt, args.random_mask, args.gt_mask),
        "cav_vs_random": (acts_cav, acts_random, args.cav_mask, args.random_mask),
    }
    cka_reports = {}
    for name, (acts_a, acts_b, mask_a, mask_b) in comparisons.items():
        result = compute_layerwise_cka(acts_a, acts_b, device="cuda" if torch.cuda.is_available() else "cpu")
        report = {
            "comparison": name,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "model_name": args.model,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "cka": {
                "mean": result["mean_cka"],
                "min": result["min_cka"],
                "max": result["max_cka"],
                "n_layers": result["n_layers"],
                "n_skipped": result.get("n_skipped", 0),
            },
            "per_layer_cka": result["per_layer"],
        }
        cka_reports[name] = report
        summarize_cka(name, report["cka"])
        write_json(os.path.join(args.output_dir, f"cka_{name}.json"), report)

    layer_names = sorted(
        set(eranks["Cold-CAV-DPO"]) | set(eranks["Oracle-DPO"]) | set(eranks["Random-DPO"]),
        key=lambda n: (layer_index(n), n),
    )
    csv_path = os.path.join(args.output_dir, "cav_gt_random_layer_diagnostics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer", "layer_index",
                "cav_density", "gt_density", "random_density",
                "cav_erank", "gt_erank", "random_erank",
                "cav_erank_norm", "gt_erank_norm", "random_erank_norm",
                "cka_cav_gt", "cka_random_gt", "cka_cav_random",
            ],
        )
        writer.writeheader()
        for lname in layer_names:
            row = {
                "layer": lname,
                "layer_index": layer_index(lname),
            }
            for key, prefix in [
                ("Cold-CAV-DPO", "cav"),
                ("Oracle-DPO", "gt"),
                ("Random-DPO", "random"),
            ]:
                stats = eranks[key].get(lname, {})
                row[f"{prefix}_density"] = stats.get("density")
                row[f"{prefix}_erank"] = stats.get("erank")
                row[f"{prefix}_erank_norm"] = stats.get("erank_norm")
            row["cka_cav_gt"] = cka_reports["cav_vs_gt"]["per_layer_cka"].get(lname)
            row["cka_random_gt"] = cka_reports["random_vs_gt"]["per_layer_cka"].get(lname)
            row["cka_cav_random"] = cka_reports["cav_vs_random"]["per_layer_cka"].get(lname)
            writer.writerow(row)
            print(
                f"[table] L{row['layer_index']:02d} "
                f"erank(cav/gt/rand)="
                f"{row['cav_erank']:.1f}/{row['gt_erank']:.1f}/{row['random_erank']:.1f} "
                f"cka(cav-gt/rand-gt/cav-rand)="
                f"{row['cka_cav_gt']}/{row['cka_random_gt']}/{row['cka_cav_random']}",
                flush=True,
            )

    summary = {
        "masks": {
            "Cold-CAV-DPO": args.cav_mask,
            "Oracle-DPO": args.gt_mask,
            "Random-DPO": args.random_mask,
        },
        "metadata": {
            "cav": cav_meta,
            "gt": gt_meta,
            "random": random_meta,
        },
        "cka": {k: v["cka"] for k, v in cka_reports.items()},
        "csv": csv_path,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(f"[out] CSV     -> {csv_path}")
    print(f"[out] summary -> {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare cold-start masks by collecting activations and reporting layer-wise CKA."""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.cold_start.utils.activation_hooks import FeatureExtractor, infer_model_input_device

DEFAULT_CALIBRATION_DATASET = "qihoo360/Light-R1-DPOData"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _msg_to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("value", "")
    if isinstance(x, list):
        return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
    return str(x)


def load_calibration_samples(
    n_samples=64,
    seed=42,
    dataset_name: str = DEFAULT_CALIBRATION_DATASET,
):
    """Return (chosen_texts, rejected_texts) lists of length n_samples."""
    print(f"Loading {n_samples} calibration samples from {dataset_name}...")
    raw = load_dataset(dataset_name, split="train").shuffle(seed=seed)

    chosen_texts, rejected_texts = [], []
    for rec in raw:
        if len(chosen_texts) >= n_samples:
            break
        prompt_raw = rec.get("prompt", "")
        if isinstance(prompt_raw, list):
            prompt = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            prompt = _msg_to_text(prompt_raw).strip()

        chosen   = _msg_to_text(rec.get("chosen",   "")).strip()
        rejected = _msg_to_text(rec.get("rejected", "")).strip()
        if not chosen or not rejected:
            continue

        chosen_texts.append(prompt + "\n" + chosen)
        rejected_texts.append(prompt + "\n" + rejected)

    print(f"  Loaded {len(chosen_texts)} samples.")
    return chosen_texts, rejected_texts


def load_masks(path):
    """Load a masks dict from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], data.get("metadata")
    if isinstance(data, dict):
        return data, None
    raise ValueError(f"Unrecognized mask format: {path}")


def apply_mask(model, masks):
    """Apply a mask in-place and return a snapshot for later restore."""
    original = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                original[name] = param.data.clone()
                param.data.mul_(masks[name].to(param.device).float())
    return original


def restore_weights(model, original):
    """Restore model weights from a snapshot returned by `apply_mask`."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original:
                param.data.copy_(original[name])

def _hsic(Kc: torch.Tensor, Lc: torch.Tensor, n: int) -> torch.Tensor:
    """Biased HSIC estimator on pre-centered Gram matrices."""
    return (Kc * Lc).sum() / (n - 1) ** 2


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two activation matrices."""
    X = X.double()
    Y = Y.double()
    n = X.shape[0]

    if n < 2:
        return float("nan")

    K = X @ X.t()   # [n, n]
    L = Y @ Y.t()   # [n, n]

    ones = torch.ones(n, n, dtype=X.dtype, device=X.device) / n
    H    = torch.eye(n,     dtype=X.dtype, device=X.device) - ones

    Kc = H @ K @ H   # [n, n]
    Lc = H @ L @ H   # [n, n]

    hsic_kl = _hsic(Kc, Lc, n)
    hsic_kk = _hsic(Kc, Kc, n)
    hsic_ll = _hsic(Lc, Lc, n)

    denom = (hsic_kk * hsic_ll).sqrt()
    if denom.abs().item() < 1e-30:
        return 0.0
    return (hsic_kl / denom).clamp(0.0, 1.0).item()

def collect_activations(model, extractor, tokenizer, texts, device,
                        max_length=512, batch_size=4):
    """Run forward passes and return pooled MLP activations per layer."""
    return extractor.collect(
        model, tokenizer, texts, device,
        max_length=max_length, batch_size=batch_size,
    )


def compute_layerwise_cka(acts_a: dict, acts_b: dict, device="cpu") -> dict:
    """Compute linear CKA for each shared layer in two activation dicts."""
    common = sorted(set(acts_a.keys()) & set(acts_b.keys()))
    if not common:
        return {"mean_cka": 0.0, "min_cka": 0.0, "max_cka": 0.0,
                "per_layer": {}, "n_layers": 0, "n_skipped": 0,
                "note": "No common layer names between the two activation dicts."}

    per_layer = {}
    n_skipped = 0
    for name in common:
        X = acts_a[name].to(device)
        Y = acts_b[name].to(device)

        if X.shape[0] != Y.shape[0]:
            per_layer[name] = None
            n_skipped += 1
            continue

        score = linear_cka(X, Y)
        per_layer[name] = None if score != score else round(score, 6)  # nan -> None

    valid = [v for v in per_layer.values() if v is not None]
    if not valid:
        all_skipped = sum(v is None for v in per_layer.values())
        return {"mean_cka": 0.0, "min_cka": 0.0, "max_cka": 0.0,
                "per_layer": per_layer, "n_layers": 0,
                "n_skipped": all_skipped}

    return {
        "mean_cka": round(sum(valid) / len(valid), 6),
        "min_cka":  round(min(valid), 6),
        "max_cka":  round(max(valid), 6),
        "per_layer": per_layer,
        "n_layers":  len(valid),
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# CLI
#
# Pseudocode (main):
#
#   parse arguments
#   validate mask file paths
#   set_seed; load masks A and B; load model + tokenizer
#   load_calibration_samples -> chosen_texts, rejected_texts
#   register FeatureExtractor hooks on model
#
#   if compare == "mask_vs_mask":
#       apply mask A  -> collect acts_a  -> restore
#       apply mask B  -> collect acts_b  -> restore
#
#   elif compare == "original_vs_a":
#       collect acts_a (no mask)
#       apply mask A  -> collect acts_b  -> restore
#
#   elif compare == "original_vs_b":
#       collect acts_a (no mask)
#       apply mask B  -> collect acts_b  -> restore
#
#   elif compare == "chosen_vs_rejected":
#       apply mask A
#       collect acts_a on chosen_texts
#       collect acts_b on rejected_texts
#       restore
#
#   remove hooks
#   result = compute_layerwise_cka(acts_a, acts_b)
#   build report dict; write JSON; print summary
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise linear CKA comparison tool for cold-start masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mask_a", type=str, help="First mask file (.pt)")
    parser.add_argument("mask_b", type=str, help="Second mask file (.pt)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m-it",
        help=(
            "HF model id used for activations. Must match the architecture the masks were "
            "built for (same layer/param names as in the .pt files)."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_CALIBRATION_DATASET,
        help="HF dataset id for calibration prompts (chosen/rejected).",
    )
    parser.add_argument(
        "--compare", type=str, default="mask_vs_mask",
        choices=["mask_vs_mask", "original_vs_a", "original_vs_b", "chosen_vs_rejected"],
        help=(
            "What to compare:\n"
            "  mask_vs_mask       - subnetwork A activations vs subnetwork B activations\n"
            "  original_vs_a      - full model vs subnetwork A\n"
            "  original_vs_b      - full model vs subnetwork B\n"
            "  chosen_vs_rejected - mask_a subnetwork on chosen vs rejected texts"
        ),
    )
    parser.add_argument("--n_samples",  type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path. Auto-generated from mask filenames if omitted.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    for path in (args.mask_a, args.mask_b):
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    set_seed(args.seed)

    # ---- Load masks -------------------------------------------------------
    print("Loading mask A:", args.mask_a)
    masks_a, meta_a = load_masks(args.mask_a)
    print("Loading mask B:", args.mask_b)
    masks_b, meta_b = load_masks(args.mask_b)

    # ---- Load model -------------------------------------------------------
    device = torch.device(args.device)
    print("\nLoading model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.device == "cpu":
        model.to(device)
    model.config.use_cache = False
    model.eval()
    input_device = infer_model_input_device(model)

    # ---- Load calibration data --------------------------------------------
    chosen_texts, rejected_texts = load_calibration_samples(
        args.n_samples,
        seed=args.seed,
        dataset_name=args.dataset_name,
    )

    # ---- Collect activations for both conditions --------------------------
    extractor = FeatureExtractor()
    extractor.register(model)

    compare = args.compare
    print(f"\n[CKA] Compare mode: {compare}")

    if compare == "mask_vs_mask":
        print("[CKA] Collecting activations for subnetwork A...")
        orig_a = apply_mask(model, masks_a)
        acts_a = collect_activations(model, extractor, tokenizer, chosen_texts,
                                     input_device, args.max_length, args.batch_size)
        restore_weights(model, orig_a)

        print("[CKA] Collecting activations for subnetwork B...")
        orig_b = apply_mask(model, masks_b)
        acts_b = collect_activations(model, extractor, tokenizer, chosen_texts,
                                     input_device, args.max_length, args.batch_size)
        restore_weights(model, orig_b)
        label_a, label_b = "mask_a_subnetwork", "mask_b_subnetwork"

    elif compare == "original_vs_a":
        print("[CKA] Collecting activations for original model...")
        acts_a = collect_activations(model, extractor, tokenizer, chosen_texts,
                                     input_device, args.max_length, args.batch_size)

        print("[CKA] Collecting activations for subnetwork A...")
        orig_a = apply_mask(model, masks_a)
        acts_b = collect_activations(model, extractor, tokenizer, chosen_texts,
                                     input_device, args.max_length, args.batch_size)
        restore_weights(model, orig_a)
        label_a, label_b = "original", "mask_a_subnetwork"

    elif compare == "original_vs_b":
        print("[CKA] Collecting activations for original model...")
        acts_a = collect_activations(model, extractor, tokenizer, chosen_texts,
                                     input_device, args.max_length, args.batch_size)

        print("[CKA] Collecting activations for subnetwork B...")
        orig_b = apply_mask(model, masks_b)
        acts_b = collect_activations(model, extractor, tokenizer, chosen_texts,
                                     input_device, args.max_length, args.batch_size)
        restore_weights(model, orig_b)
        label_a, label_b = "original", "mask_b_subnetwork"

    elif compare == "chosen_vs_rejected":
        print("[CKA] Applying subnetwork A mask...")
        orig_a = apply_mask(model, masks_a)

        print("[CKA] Collecting chosen activations...")
        acts_a = collect_activations(model, extractor, tokenizer, chosen_texts,
                                     input_device, args.max_length, args.batch_size)
        print("[CKA] Collecting rejected activations...")
        acts_b = collect_activations(model, extractor, tokenizer, rejected_texts,
                                     input_device, args.max_length, args.batch_size)
        restore_weights(model, orig_a)
        label_a, label_b = "chosen", "rejected"

    else:
        raise ValueError(f"Unknown compare mode: {compare}")

    extractor.remove()

    # ---- Compute layer-wise CKA -------------------------------------------
    print(f"\n[CKA] Computing layer-wise linear CKA (n={args.n_samples} samples)...")
    result = compute_layerwise_cka(acts_a, acts_b, device=args.device)

    # ---- Build report -----------------------------------------------------
    report = {
        "mask_a": os.path.abspath(args.mask_a),
        "mask_b": os.path.abspath(args.mask_b),
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "compare": compare,
        "label_a": label_a,
        "label_b": label_b,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "cka": {
            "mean": result["mean_cka"],
            "min":  result["min_cka"],
            "max":  result["max_cka"],
            "n_layers":  result.get("n_layers"),
            "n_skipped": result.get("n_skipped", 0),
        },
        "per_layer_cka": result["per_layer"],
    }
    if "note" in result:
        report["note"] = result["note"]
    if meta_a:
        report["metadata_a"] = meta_a
    if meta_b:
        report["metadata_b"] = meta_b

    # ---- Output path ------------------------------------------------------
    if args.output is None:
        base_a = os.path.splitext(os.path.basename(args.mask_a))[0]
        base_b = os.path.splitext(os.path.basename(args.mask_b))[0]
        try:
            out_dir = os.path.commonpath(
                [os.path.abspath(args.mask_a), os.path.abspath(args.mask_b)]
            )
        except ValueError:
            out_dir = "."
        args.output = os.path.join(out_dir, f"cka_{compare}_{base_a}_vs_{base_b}.json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nCKA summary  ({label_a}  vs  {label_b}):")
    print(f"  mean: {result['mean_cka']:.4f}")
    print(f"  min:  {result['min_cka']:.4f}")
    print(f"  max:  {result['max_cka']:.4f}")
    if result.get("n_skipped", 0):
        print(f"  skipped (shape mismatch): {result['n_skipped']} layers")
    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()

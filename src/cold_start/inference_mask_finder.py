#!/usr/bin/env python3
"""
Cold-Start Inference Mask Finder
=================================
Generates a sparse subnetwork mask WITHOUT any training steps.

Supported methods:
  cav   – Concept Activation Vectors (Chosen vs Rejected probes)   [default]
  snip  – SNIP gradient saliency (single backward pass, no update)

Output format is identical to even_better_mask_finder.py and is directly
compatible with SparseMaskManager:
  torch.save({"masks": masks, "metadata": metadata}, path)

Usage:
  python src/cold_start/inference_mask_finder.py \
      --method cav \
      --n_samples 64 \
      --sparsity 90.0 \
      --output masks/cold_start_cav_90pct.pt

  python src/cold_start/inference_mask_finder.py \
      --method snip \
      --n_samples 64 \
      --sparsity 90.0 \
      --output masks/cold_start_snip_90pct.pt
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure project root is on the path regardless of CWD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.cold_start.utils.activation_hooks import FeatureExtractor
from src.cold_start.utils.cav_probes import CAVProbeScorer
from src.cold_start.utils.snip_scorer import SNIPScorer


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

DATASET_NAME = "qihoo360/Light-R1-DPOData"


def _msg_to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("value", "")
    if isinstance(x, list):
        return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
    return str(x)


def load_calibration_samples(n_samples=64):
    """
    Load n_samples from the DPO dataset.
    Returns lists: chosen_texts, rejected_texts
    """
    print(f"Loading {n_samples} calibration samples from {DATASET_NAME}...")
    raw = load_dataset(DATASET_NAME, split="train")

    chosen_texts   = []
    rejected_texts = []

    for rec in raw:
        if len(chosen_texts) >= n_samples:
            break

        prompt_raw   = rec.get("prompt", "")
        chosen_raw   = rec.get("chosen", "")
        rejected_raw = rec.get("rejected", "")

        if isinstance(prompt_raw, list):
            prompt = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            prompt = _msg_to_text(prompt_raw).strip()

        chosen   = _msg_to_text(chosen_raw).strip()
        rejected = _msg_to_text(rejected_raw).strip()

        if not chosen or not rejected:
            continue

        # Concatenate prompt + response for context
        chosen_texts.append(prompt + "\n" + chosen)
        rejected_texts.append(prompt + "\n" + rejected)

    print(f"  Loaded {len(chosen_texts)} chosen / {len(rejected_texts)} rejected samples.")
    return chosen_texts, rejected_texts


# ---------------------------------------------------------------------------
# Mask saving (SparseMaskManager-compatible)
# ---------------------------------------------------------------------------

def save_masks(masks, output_path, metadata=None):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_dict = {"masks": masks}
    if metadata:
        save_dict["metadata"] = metadata
    torch.save(save_dict, output_path)

    total  = sum(m.numel() for m in masks.values())
    kept   = sum(m.sum().item() for m in masks.values())
    actual = 100.0 * (1.0 - kept / total) if total > 0 else 0.0
    print(f"\n[save_masks] Saved {len(masks)} masks → {output_path}")
    print(f"  Parameters covered : {total:,}")
    print(f"  Active (kept)      : {int(kept):,}")
    print(f"  Sparsity           : {actual:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Cold-Start Mask Finder  [method={args.method}]")
    print(f"{'='*60}")
    print(f"  Model      : {args.model_name}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  Sparsity   : {args.sparsity}%")
    print(f"  Output     : {args.output}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    # ---- Load tokenizer ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Load model (inference only) -----------------------------------
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.config.use_cache = False
    print(f"  Model on {next(model.parameters()).device}, dtype={model.dtype}")

    # ---- Load calibration data -----------------------------------------
    chosen_texts, rejected_texts = load_calibration_samples(args.n_samples)

    # ---- Run scoring ---------------------------------------------------
    if args.method == "cav":
        extractor = FeatureExtractor()
        extractor.register(model)

        print("\n[CAV] Collecting chosen activations...")
        pos_acts = extractor.collect(
            model, tokenizer, chosen_texts, device,
            max_length=args.max_length, batch_size=args.batch_size,
        )

        print("[CAV] Collecting rejected activations...")
        neg_acts = extractor.collect(
            model, tokenizer, rejected_texts, device,
            max_length=args.max_length, batch_size=args.batch_size,
        )

        extractor.remove()

        print("[CAV] Training linear probes...")
        scorer = CAVProbeScorer()
        neuron_scores = scorer.score(pos_acts, neg_acts, mag_weight=args.mag_weight)
        masks = scorer.scores_to_masks(neuron_scores, model, sparsity_percent=args.sparsity)

        metadata = {
            "method": "cav",
            "model_name": args.model_name,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": DATASET_NAME,
            "mag_weight": args.mag_weight,
        }

    elif args.method == "snip":
        print("[SNIP] Computing gradient saliency scores...")
        scorer = SNIPScorer()
        snip_scores = scorer.score(
            model, tokenizer, chosen_texts, device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        masks = scorer.scores_to_masks(snip_scores, sparsity_percent=args.sparsity)

        metadata = {
            "method": "snip",
            "model_name": args.model_name,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": DATASET_NAME,
        }

    else:
        raise ValueError(f"Unknown method: {args.method}. Choose 'cav' or 'snip'.")

    # ---- Save mask -----------------------------------------------------
    save_masks(masks, args.output, metadata)
    print("\n[Done] Cold-start mask generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cold-start inference-only mask finder (CAV / SNIP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument(
        "--method", type=str, default="cav", choices=["cav", "snip"],
        help="Scoring method: cav (linear probes) or snip (gradient saliency)",
    )
    parser.add_argument("--n_samples", type=int, default=64,
                        help="Number of calibration DPO pairs")
    parser.add_argument("--sparsity", type=float, default=90.0,
                        help="Target sparsity %%: percentage of weights zeroed out")
    parser.add_argument("--output", type=str, default="masks/cold_start_cav_90pct.pt",
                        help="Output .pt file path")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--mag_weight", type=float, default=1.0,
        help=(
            "[CAV only] Weight of task-presence signal (mean |act| on chosen) "
            "relative to the discriminative CAV signal. "
            "0.0 = pure discriminative (original); 1.0 = equal blend (default). "
            "Higher values retain more shared-subnetwork weights."
        ),
    )

    main(parser.parse_args())

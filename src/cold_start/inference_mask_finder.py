#!/usr/bin/env python3
"""
Unified cold-start mask finder.

Supports four scoring methods:
  --method fisher      Diagonal Fisher Information (empirical, mini-batch smoothed)
  --method cav         Linear-probe (CAV) discriminative scores
  --method snip        Gradient saliency (|grad * weight|)
  --method activation  Mean |activation| magnitude

Data is loaded as text pairs from HuggingFace via --mode dpo|grpo.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.mask_utils import (
    create_mask_from_scores_gpu_efficient,
    compute_jaccard_similarity,
    save_masks,
)
from src.cold_start.utils.activation_hooks import FeatureExtractor, infer_model_input_device
from src.cold_start.utils.cav_probes import CAVProbeScorer
from src.cold_start.utils.snip_scorer import SNIPScorer

# ── Default datasets ──────────────────────────────────────────
DPO_DATASET_NAME = "qihoo360/Light-R1-DPOData"
GRPO_DATASET_NAME = "open-r1/OpenR1-Math-220k"


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _msg_to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("value", "")
    if isinstance(x, list):
        return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
    return str(x)


# ══════════════════════════════════════════════════════════════
#  Data loading  (text-list based, shared by all methods)
# ══════════════════════════════════════════════════════════════

def load_calibration_samples_dpo(dataset_name, n_samples=64, seed=42):
    """Load chosen/rejected pairs from a DPO-format dataset."""
    print(f"[DPO mode] Loading {n_samples} calibration samples from {dataset_name}...")
    raw = load_dataset(dataset_name, split="train")
    raw = raw.shuffle(seed=seed)

    chosen_texts: List[str] = []
    rejected_texts: List[str] = []

    for rec in raw:
        if len(chosen_texts) >= n_samples:
            break

        prompt_raw = rec.get("prompt", "")
        chosen_raw = rec.get("chosen", "")
        rejected_raw = rec.get("rejected", "")

        if not prompt_raw:
            convs = rec.get("conversations", [])
            if isinstance(convs, list):
                human_parts = [
                    m.get("value", "") for m in convs
                    if isinstance(m, dict)
                    and m.get("from", m.get("role", "")).lower() in ("human", "user", "system")
                ]
                prompt_raw = "\n".join(human_parts).strip()

        if isinstance(prompt_raw, list):
            prompt = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            prompt = _msg_to_text(prompt_raw).strip()

        chosen = _msg_to_text(chosen_raw).strip()
        rejected = _msg_to_text(rejected_raw).strip()

        if not chosen or not rejected:
            continue

        chosen_texts.append(prompt + "\n" + chosen)
        rejected_texts.append(prompt + "\n" + rejected)

    print(f"  Loaded {len(chosen_texts)} chosen / {len(rejected_texts)} rejected samples.")
    return chosen_texts, rejected_texts


def load_calibration_samples_grpo(dataset_name, n_samples=64, seed=42):
    """Load problem/solution pairs from a GRPO-format dataset."""
    print(f"[GRPO mode] Loading {n_samples} calibration samples from {dataset_name}...")
    raw = load_dataset(dataset_name, split="train")
    raw = raw.shuffle(seed=seed)

    INPUT_CANDIDATES = ["problem", "query", "prompt", "input", "text"]
    OUTPUT_CANDIDATES = ["solution", "response", "completion", "answer", "output"]

    cols = raw.column_names
    input_col = next((c for c in INPUT_CANDIDATES if c in cols), None)
    output_col = next((c for c in OUTPUT_CANDIDATES if c in cols), None)

    if input_col is None or output_col is None:
        raise KeyError(
            f"Could not find input/output columns in GRPO dataset '{dataset_name}'. "
            f"Available: {cols}. "
            f"Tried inputs={INPUT_CANDIDATES}, outputs={OUTPUT_CANDIDATES}."
        )

    positive_texts: List[str] = []
    negative_texts: List[str] = []

    for rec in raw:
        if len(positive_texts) >= n_samples:
            break

        prompt = str(rec.get(input_col, "") or "").strip()
        solution = str(rec.get(output_col, "") or "").strip()

        if not prompt or not solution:
            continue

        positive_texts.append(prompt + "\n" + solution)
        negative_texts.append(prompt)

    print(f"  Loaded {len(positive_texts)} positive (prompt+solution) / "
          f"{len(negative_texts)} negative (prompt-only) samples.")
    return positive_texts, negative_texts


def load_calibration_samples(n_samples=64, seed=42, mode="dpo", dataset_name=None):
    """Dispatch to the appropriate loader based on training mode."""
    if mode == "dpo":
        ds = dataset_name or DPO_DATASET_NAME
        return load_calibration_samples_dpo(ds, n_samples=n_samples, seed=seed)
    elif mode == "grpo":
        ds = dataset_name or GRPO_DATASET_NAME
        return load_calibration_samples_grpo(ds, n_samples=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'dpo' or 'grpo'.")


# ══════════════════════════════════════════════════════════════
#  Fisher scoring
# ══════════════════════════════════════════════════════════════

def _encode_for_fisher(tokenizer, chosen_texts, max_length, device):
    """Tokenise chosen texts with prompt-masked labels for Fisher scoring."""
    pairs = []
    for text in chosen_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        labels = enc["input_ids"].clone()
        pairs.append({
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "labels": labels.to(device),
        })
    return pairs


def compute_fisher_scores(model, calibration_data, device="cuda", mlp_only=True,
                          mini_batch_size=4, normalize_per_layer=True):
    """Compute diagonal Fisher Information scores.

    Uses empirical Fisher with mini-batch gradient smoothing and
    optional per-layer z-score normalisation.
    """
    print(f"\n=== Computing Improved Empirical Fisher Information ===")
    print(f"  Calibration samples:       {len(calibration_data)}")
    print(f"  MLP parameters only:       {mlp_only}")
    print(f"  Mini-batch size:           {mini_batch_size}")
    print(f"  Per-layer normalization:   {normalize_per_layer}")

    model.eval()

    MLP_KEYWORDS = ["gate_proj", "up_proj", "down_proj", "fc1", "fc2",
                    "feed_forward", "ffn", "mlp.c_fc", "mlp.c_proj"]

    def is_mlp_param(name):
        return any(kw in name.lower() for kw in MLP_KEYWORDS)

    fisher_scores = {}
    scored_params = {}
    for name, param in model.named_parameters():
        if mlp_only and not is_mlp_param(name):
            continue
        if not param.requires_grad:
            continue
        fisher_scores[name] = torch.zeros_like(param, dtype=torch.float32)
        scored_params[name] = param

    print(f"  Scoring {len(fisher_scores)} parameter tensors")

    n_batches = 0
    batch_start = 0
    total_samples = len(calibration_data)

    while batch_start < total_samples:
        batch_end = min(batch_start + mini_batch_size, total_samples)
        batch = calibration_data[batch_start:batch_end]
        batch_size = len(batch)

        if batch_start % (mini_batch_size * 10) == 0:
            print(f"  [{batch_start+1}/{total_samples}] Processing mini-batch...")

        model.zero_grad()

        total_loss = None
        for enc in batch:
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=enc["labels"],
            )
            loss = outputs.loss / batch_size
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        total_loss.backward()

        with torch.no_grad():
            for name, param in scored_params.items():
                if param.grad is not None:
                    fisher_scores[name] += param.grad.float().pow(2)

        n_batches += 1
        batch_start = batch_end

    print(f"\n  Normalizing by {n_batches} mini-batches ({total_samples} samples)...")
    for name in fisher_scores:
        fisher_scores[name] /= n_batches

    if normalize_per_layer:
        print("  Applying per-layer z-score normalization...")
        for name in fisher_scores:
            s = fisher_scores[name]
            mean = s.mean()
            std = s.std()
            if std > 1e-12:
                fisher_scores[name] = (s - mean) / std
            else:
                fisher_scores[name] = torch.zeros_like(s)

    all_scores = torch.cat([s.flatten() for s in fisher_scores.values()])
    print(f"  Fisher score stats (after normalization):")
    print(f"    min:  {all_scores.min().item():.6e}")
    print(f"    max:  {all_scores.max().item():.6e}")
    print(f"    mean: {all_scores.mean().item():.6e}")
    print(f"    std:  {all_scores.std().item():.6e}")
    print(f"    nonzero fraction: {(all_scores != 0).float().mean().item():.4f}")

    return fisher_scores


# ══════════════════════════════════════════════════════════════
#  CAV diagnostics  (only activated with --verbose)
# ══════════════════════════════════════════════════════════════

def summarize_layer_score_distribution(scores: Dict[str, torch.Tensor], top_n_preview: int = 8) -> Dict[str, object]:
    rows: List[Dict[str, float]] = []
    for name, t in scores.items():
        if t is None or t.numel() == 0:
            continue
        s = t.detach().float().reshape(-1).cpu()
        rows.append({
            "layer": name,
            "mean": float(s.mean().item()),
            "std": float(s.std(unbiased=False).item()),
            "min": float(s.min().item()),
            "max": float(s.max().item()),
            "numel": float(s.numel()),
        })

    rows = sorted(rows, key=lambda x: x["layer"])
    print("\n=== Per-layer score distribution (before masking) ===")
    for r in rows:
        print(f"  {r['layer']}: mean={r['mean']:.6e}, std={r['std']:.6e}, "
              f"min={r['min']:.6e}, max={r['max']:.6e}, n={int(r['numel'])}")

    if not rows:
        return {"n_layers": 0, "mean_std": 0.0, "max_std": 0.0, "min_std": 0.0,
                "near_zero_std_layers": 0, "weak_signal": True}

    stds = [r["std"] for r in rows]
    mins = [r["min"] for r in rows]
    maxs = [r["max"] for r in rows]

    preview = sorted(rows, key=lambda x: x["std"])[:max(1, top_n_preview)]
    print("\nLowest-variance layers (possible CAV degeneracy):")
    for r in preview:
        print(f"  {r['layer']}: std={r['std']:.6e}, range=({r['min']:.6e}, {r['max']:.6e})")

    near_zero_std_layers = sum(1 for s in stds if s <= 1e-10)
    weak_signal = (max(stds) <= 1e-8) or (near_zero_std_layers == len(stds))
    if weak_signal:
        print("\n[warn] CAV score variance is near-zero across layers. "
              "This suggests weak cold-start signal or score collapse.")

    return {
        "n_layers": len(rows), "mean_std": float(sum(stds) / len(stds)),
        "max_std": float(max(stds)), "min_std": float(min(stds)),
        "global_min": float(min(mins)), "global_max": float(max(maxs)),
        "near_zero_std_layers": int(near_zero_std_layers), "weak_signal": bool(weak_signal),
    }


def maybe_add_score_noise(scores: Dict[str, torch.Tensor], noise_ratio: float) -> Dict[str, torch.Tensor]:
    if noise_ratio <= 0:
        return scores
    out: Dict[str, torch.Tensor] = {}
    print(f"Adding CAV tie-break noise to scores (ratio={noise_ratio:.3e})...")
    for name, t in scores.items():
        s = t.detach().float().cpu()
        scale = max(float(s.abs().max().item()) * noise_ratio, 1e-12)
        out[name] = s + torch.randn_like(s) * scale
    return out


def summarize_mask_sparsity(masks: Dict[str, torch.Tensor], top_n_preview: int = 8) -> Dict[str, object]:
    rows = []
    for name, m in masks.items():
        mm = m.detach().float().cpu()
        total = int(mm.numel())
        kept = float(mm.sum().item())
        sparsity = 1.0 - (kept / total if total > 0 else 0.0)
        rows.append({"layer": name, "sparsity": float(sparsity), "total": total})

    rows = sorted(rows, key=lambda x: x["layer"])
    print("\n=== Per-layer mask sparsity ===")
    for r in rows:
        print(f"  {r['layer']}: sparsity={r['sparsity']:.6f} (n={r['total']})")

    if not rows:
        return {"n_layers": 0}

    sparsities = [r["sparsity"] for r in rows]
    print(f"Sparsity range across layers: min={min(sparsities):.6f}, "
          f"max={max(sparsities):.6f}, span={max(sparsities) - min(sparsities):.6f}")

    flat_preview = sorted(rows, key=lambda x: x["sparsity"])[:max(1, top_n_preview)]
    print("Lowest sparsity layers preview:")
    for r in flat_preview:
        print(f"  {r['layer']}: {r['sparsity']:.6f}")

    return {
        "n_layers": len(rows),
        "min_sparsity": float(min(sparsities)),
        "max_sparsity": float(max(sparsities)),
        "mean_sparsity": float(sum(sparsities) / len(sparsities)),
        "span": float(max(sparsities) - min(sparsities)),
    }


def effective_rank_normalized(mask_tensor: torch.Tensor, eps: float = 1e-12) -> Optional[float]:
    if mask_tensor.ndim < 2:
        return None
    W = mask_tensor.detach().float().reshape(mask_tensor.shape[0], -1).cpu()
    m, n = W.shape
    max_rank = min(m, n)
    if max_rank <= 0:
        return 0.0
    s = torch.linalg.svdvals(W)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum().item()
    erank = float(math.exp(entropy))
    return erank / float(max_rank)


def summarize_effective_rank(masks: Dict[str, torch.Tensor], threshold: float = 0.3) -> Dict[str, object]:
    rows = []
    for name, m in masks.items():
        val = effective_rank_normalized(m)
        if val is None:
            continue
        rows.append({"layer": name, "erank_norm": float(val)})

    rows = sorted(rows, key=lambda x: x["layer"])
    print("\n=== Per-layer normalized effective rank (post-mask) ===")
    for r in rows:
        print(f"  {r['layer']}: erank_norm={r['erank_norm']:.6f}")

    if not rows:
        return {"n_layers": 0, "fraction_above_threshold": 0.0, "threshold": threshold}

    values = [r["erank_norm"] for r in rows]
    above = sum(1 for v in values if v > threshold)
    print(f"Effective-rank summary: mean={sum(values)/len(values):.6f}, "
          f"min={min(values):.6f}, max={max(values):.6f}, "
          f"layers_above_{threshold:.2f}={above}/{len(values)}")
    if above < max(1, int(0.5 * len(values))):
        print("[warn] Effective rank remains low in many layers. "
              "Likely cause: degenerate/low-variance CAV scores in cold start.")

    return {
        "n_layers": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "threshold": float(threshold),
        "num_above_threshold": int(above),
        "fraction_above_threshold": float(above / len(values)),
    }


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"\n{'='*60}")
    print(f"Cold-Start Mask Finder  [method={args.method}  mode={args.mode}]")
    print(f"{'='*60}")
    print(f"  Model      : {args.model_name}")
    print(f"  Mode       : {args.mode}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  Sparsity   : {args.sparsity}%")
    print(f"  Output     : {args.output}")
    print(f"  Seed       : {args.seed}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() and not args.force_cpu else None,
    )
    if device == "cpu":
        model.to(device)
    model.config.use_cache = False
    input_device = infer_model_input_device(model)
    print(f"  Model on {next(model.parameters()).device}, dtype={model.dtype}")
    print(f"  Input device: {input_device}")

    chosen_texts, rejected_texts = load_calibration_samples(
        args.n_samples, seed=args.seed, mode=args.mode, dataset_name=args.dataset_name,
    )
    effective_dataset = args.dataset_name or (
        DPO_DATASET_NAME if args.mode == "dpo" else GRPO_DATASET_NAME
    )

    # ── Dispatch by method ────────────────────────────────────
    score_variance_summary = None

    if args.method == "fisher":
        calibration_data = _encode_for_fisher(tokenizer, chosen_texts, args.max_length, input_device)
        score_dict = compute_fisher_scores(
            model, calibration_data, device=device,
            mlp_only=True,
            mini_batch_size=args.mini_batch_size,
            normalize_per_layer=not args.no_layer_norm,
        )
        masks = create_mask_from_scores_gpu_efficient(
            score_dict, args.sparsity, device=device,
            local_pool=args.local_pool,
        )
        metadata = {
            "method": "fisher",
            "model_name": args.model_name,
            "mode": args.mode,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": effective_dataset,
            "seed": args.seed,
            "mini_batch_size": args.mini_batch_size,
            "layer_normalization": not args.no_layer_norm,
        }

    elif args.method == "cav":
        extractor = FeatureExtractor()
        extractor.register(model)

        print("\n[CAV] Collecting chosen activations...")
        pos_acts = extractor.collect(
            model, tokenizer, chosen_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        print("[CAV] Collecting rejected activations...")
        neg_acts = extractor.collect(
            model, tokenizer, rejected_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        extractor.remove()

        print("[CAV] Training linear probes...")
        scorer = CAVProbeScorer()
        neuron_scores = scorer.score(pos_acts, neg_acts, mag_weight=args.mag_weight)
        masks = scorer.scores_to_masks(
            neuron_scores, model,
            sparsity_percent=args.sparsity,
            local_pool=args.local_pool,
        )
        metadata = {
            "method": "cav",
            "model_name": args.model_name,
            "mode": args.mode,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": effective_dataset,
            "seed": args.seed,
            "mag_weight": args.mag_weight,
            "local_pool": args.local_pool,
        }

        if args.verbose:
            score_dict_for_diag = {}
            for name, s in neuron_scores.items():
                score_dict_for_diag[name] = s
            score_variance_summary = summarize_layer_score_distribution(score_dict_for_diag)

    elif args.method == "snip":
        print("[SNIP] Computing gradient saliency scores...")
        scorer = SNIPScorer()
        snip_scores = scorer.score(
            model, tokenizer, chosen_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        masks = scorer.scores_to_masks(
            snip_scores, sparsity_percent=args.sparsity, local_pool=args.local_pool,
        )
        metadata = {
            "method": "snip",
            "model_name": args.model_name,
            "mode": args.mode,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": effective_dataset,
            "seed": args.seed,
        }

    elif args.method == "activation":
        extractor = FeatureExtractor()
        extractor.register(model)

        print("\n[Activation] Collecting chosen activations...")
        pos_acts = extractor.collect(
            model, tokenizer, chosen_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        print("[Activation] Collecting rejected activations...")
        neg_acts = extractor.collect(
            model, tokenizer, rejected_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        extractor.remove()

        # Score neurons by mean |activation| across chosen + rejected
        neuron_scores: Dict[str, torch.Tensor] = {}
        for name in sorted(set(pos_acts) | set(neg_acts)):
            parts = []
            if name in pos_acts:
                parts.append(pos_acts[name])
            if name in neg_acts:
                parts.append(neg_acts[name])
            combined = torch.cat(parts, dim=0)
            neuron_scores[name] = combined.abs().mean(dim=0)

        scorer = CAVProbeScorer()
        masks = scorer.scores_to_masks(
            neuron_scores, model,
            sparsity_percent=args.sparsity,
            local_pool=args.local_pool,
        )
        metadata = {
            "method": "activation",
            "model_name": args.model_name,
            "mode": args.mode,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": effective_dataset,
            "seed": args.seed,
        }

    else:
        raise ValueError(f"Unknown method: {args.method}. Choose 'fisher', 'cav', 'snip', or 'activation'.")

    # ── Verbose diagnostics ───────────────────────────────────
    if args.verbose:
        summarize_mask_sparsity(masks)
        summarize_effective_rank(masks, threshold=0.3)

    # ── Reference mask comparison ─────────────────────────────
    jaccard_results = None
    if args.reference_mask:
        print(f"\nLoading reference mask from: {args.reference_mask}")
        ref = torch.load(args.reference_mask, map_location="cpu")
        reference_masks = ref["masks"]
        jaccard_results = compute_jaccard_similarity(masks, reference_masks)
        metadata["jaccard_similarity"] = {
            "aggregate": jaccard_results["aggregate_jaccard"],
            "mean": jaccard_results["mean_jaccard"],
        }

    if score_variance_summary is not None:
        metadata["score_variance_summary"] = score_variance_summary

    # ── Save ──────────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        model_sanitized = sanitize_model_name(args.model_name)
        output_path = f"masks/cold_{args.method}_{model_sanitized}_sparsity{args.sparsity}pct.pt"

    save_masks(masks, output_path, metadata)

    if jaccard_results:
        jaccard_file = output_path.replace(".pt", "_jaccard.json")
        with open(jaccard_file, "w") as f:
            json.dump(jaccard_results, f, indent=2)
        print(f"Detailed Jaccard results saved to: {jaccard_file}")

    print("\n[Done] Cold-start mask generation complete.")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified cold-start mask finder (fisher / cav / snip / activation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Core ──────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument(
        "--method", type=str, default="cav",
        choices=["fisher", "cav", "snip", "activation"],
        help="Scoring method.",
    )
    parser.add_argument(
        "--mode", type=str, default="dpo", choices=["dpo", "grpo"],
        help=(
            "Training mode: 'dpo' loads chosen/rejected pairs; "
            "'grpo' loads prompt+solution pairs (prompt-only as CAV negative)."
        ),
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help=(
            "HuggingFace dataset for calibration. "
            f"Defaults to '{DPO_DATASET_NAME}' (dpo) / '{GRPO_DATASET_NAME}' (grpo)."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=64,
                        help="Number of calibration pairs")
    parser.add_argument("--sparsity", type=float, default=90.0,
                        help="Target sparsity %%: percentage of weights zeroed out")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt file path (auto-generated if omitted)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--local-pool", dest="local_pool", action="store_true",
        help=(
            "Use per-layer selection instead of global cross-layer ranking. "
            "Default (off): one global threshold. "
            "With --local-pool: each layer keeps top keep_frac independently."
        ),
    )
    parser.add_argument("--reference_mask", type=str, default=None,
                        help="Optional reference mask for Jaccard similarity comparison.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-layer diagnostics (score distribution, sparsity, effective rank).")
    parser.add_argument("--no_layer_norm", action="store_true",
                        help="Disable per-layer normalization (affects fisher and cav methods).")

    # ── CAV-specific ──────────────────────────────────────────
    parser.add_argument(
        "--mag_weight", type=float, default=1.0,
        help=(
            "[CAV only] Weight of task-presence signal vs discriminative signal. "
            "0.0 = pure discriminative; 1.0 = equal blend."
        ),
    )

    # ── Fisher-specific ───────────────────────────────────────
    parser.add_argument("--mini_batch_size", type=int, default=4,
                        help="[Fisher only] Mini-batch size for gradient smoothing.")

    args = parser.parse_args()
    main(args)

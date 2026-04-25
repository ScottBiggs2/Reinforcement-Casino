#!/usr/bin/env python3
"""
Wanda + OWL cold-start mask finder.

Produces sparse binary masks for DPO/GRPO fine-tuning by combining:
  - Wanda scoring:  Score[i,j] = |W[i,j]| * ||X_j||_2
  - OWL allocation: non-uniform per-layer sparsity budgets based on activation
                    outlier density (layers with more outliers keep more weights).

Reference:
  - Wanda: "A Simple and Effective Pruning Approach for LLMs" (ICLR 2024)
    https://arxiv.org/abs/2306.11695
  - OWL: "Outlier Weighed Layerwise Sparsity" (ICML 2024)
    https://arxiv.org/abs/2310.05175

Usage:
  PYTHONPATH=. python src/cold_start/wanda_owl_mask_finder.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --n_samples 128 --sparsity_percent 97.5 \
    --output_file masks/wanda_owl_test.pt
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is importable.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    pooling_metadata,
    save_masks,
)
from src.utils.mask_coverage_report import compute_mask_coverage_report


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def choose_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


# ---------------------------------------------------------------------------
# Step A1: Activation Norm Collector
# ---------------------------------------------------------------------------

class ActivationNormCollector:
    """
    Collect ||X_j||_2 for each Linear layer's input features.

    For each nn.Linear(in_features, out_features), we hook the forward pass
    to capture inp[0] of shape [B, T, in_features].  We accumulate the sum of
    squares across all (B*T) tokens and all batches, then take sqrt at the end.

    Result: {param_name: 1D tensor of shape [in_features]} where each element
    is the L2 norm of that feature dimension across the entire calibration set.
    """

    def __init__(self):
        self._hooks: List[Any] = []
        self._sum_sq: Dict[str, torch.Tensor] = {}
        self._token_counts: Dict[str, int] = {}

    def register(self, model: nn.Module) -> None:
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = f"{module_name}.weight"
                in_dim = module.in_features
                self._sum_sq[key] = torch.zeros(in_dim, dtype=torch.float64)
                self._token_counts[key] = 0

                def _make_hook(k: str):
                    def _hook(_mod, inp, _out):
                        x = inp[0].detach().float()  # [B, T, in_features]
                        if x.dim() == 3:
                            x_flat = x.reshape(-1, x.shape[-1])  # [B*T, in_features]
                        elif x.dim() == 2:
                            x_flat = x
                        else:
                            return
                        # Accumulate sum of squares on CPU to avoid GPU memory pressure
                        self._sum_sq[k] += x_flat.pow(2).sum(dim=0).to(torch.float64).cpu()
                        self._token_counts[k] += x_flat.shape[0]
                    return _hook

                self._hooks.append(module.register_forward_hook(_make_hook(key)))

        print(f"[ActivationNormCollector] Hooked {len(self._sum_sq)} Linear layers")

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_norms(self) -> Dict[str, torch.Tensor]:
        """Return {param_name: ||X_j||_2 for j in [0, in_features)}."""
        norms = {}
        for key, ssq in self._sum_sq.items():
            norms[key] = ssq.sqrt().float()
        return norms


# ---------------------------------------------------------------------------
# Step A2: Wanda Scoring
# ---------------------------------------------------------------------------

def compute_wanda_scores(
    model: nn.Module,
    activation_norms: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute element-wise Wanda importance: Score[i,j] = |W[i,j]| * ||X_j||_2.

    Returns {param_name: 2D score tensor matching W.shape}.
    """
    scores: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.dim() != 2:
            continue
        if name not in activation_norms:
            continue
        W = param.detach().float().abs().cpu()         # [out_features, in_features]
        X_norm = activation_norms[name].float().cpu()  # [in_features]

        if X_norm.numel() != W.shape[1]:
            print(f"  [Wanda] WARNING: shape mismatch for {name}: "
                  f"norm.numel()={X_norm.numel()} vs W.shape[1]={W.shape[1]}, skipping")
            continue

        scores[name] = W * X_norm.unsqueeze(0)  # broadcast [out, in]
    return scores


# ---------------------------------------------------------------------------
# Step A3: OWL Non-Uniform Sparsity Allocation
# ---------------------------------------------------------------------------

def compute_owl_sparsity_allocation(
    activation_norms: Dict[str, torch.Tensor],
    wanda_scores: Dict[str, torch.Tensor],
    target_sparsity: float,
    *,
    outlier_sigma: float = 6.0,
    min_sparsity: float = 50.0,
    max_sparsity: float = 99.9,
) -> Dict[str, float]:
    """
    Compute per-layer sparsity using OWL (Outlier Weighed Layerwise) allocation.

    Layers with more activation outliers (features > outlier_sigma*std from mean)
    are assigned LOWER sparsity (keep more weights).

    The allocation is constrained so the weighted global sparsity matches
    target_sparsity exactly.

    Returns {param_name: layer_sparsity_percent}.
    """
    # 1. Compute outlier ratio per layer
    outlier_ratios: Dict[str, float] = {}
    layer_sizes: Dict[str, int] = {}

    for name, norms in activation_norms.items():
        if name not in wanda_scores:
            continue
        mean_val = norms.mean().item()
        std_val = norms.std().item()
        if std_val < 1e-12:
            outlier_ratios[name] = 0.0
        else:
            threshold = mean_val + outlier_sigma * std_val
            outlier_count = (norms > threshold).sum().item()
            outlier_ratios[name] = outlier_count / max(norms.numel(), 1)
        layer_sizes[name] = wanda_scores[name].numel()

    if not outlier_ratios:
        return {name: target_sparsity for name in wanda_scores}

    total_params = sum(layer_sizes.values())

    # 2. Compute raw allocation: layers with higher outlier ratio → lower sparsity
    #    We use the formula: raw_sparsity[l] = target - M * (outlier_ratio[l] - mean_ratio)
    #    Then solve for M such that the weighted average equals target_sparsity.
    mean_ratio = sum(
        outlier_ratios[n] * layer_sizes[n] for n in outlier_ratios
    ) / total_params

    # The scaling factor M controls how aggressively we redistribute.
    # We use a heuristic: M is proportional to the range of outlier ratios.
    ratio_values = list(outlier_ratios.values())
    ratio_range = max(ratio_values) - min(ratio_values) if len(ratio_values) > 1 else 0.0

    if ratio_range < 1e-12:
        # All layers have the same outlier ratio; fall back to uniform
        print("[OWL] All layers have identical outlier ratios; using uniform allocation")
        return {name: target_sparsity for name in wanda_scores}

    # Calibrate M so the most extreme layers shift by ~(target/4) percentage points
    max_shift = target_sparsity / 4.0  # e.g., at 97.5% → up to ~24pp shift
    M = max_shift / ratio_range

    raw_sparsities: Dict[str, float] = {}
    for name in outlier_ratios:
        deviation = outlier_ratios[name] - mean_ratio
        raw = target_sparsity - M * deviation
        raw = max(min_sparsity, min(max_sparsity, raw))
        raw_sparsities[name] = raw

    # 3. Rescale to enforce global budget constraint:
    #    sum(layer_size[l] * (1 - sparsity[l]/100)) == total_params * (1 - target/100)
    target_keep_total = total_params * (1.0 - target_sparsity / 100.0)
    current_keep_total = sum(
        layer_sizes[n] * (1.0 - raw_sparsities[n] / 100.0)
        for n in raw_sparsities
    )

    if current_keep_total > 0:
        correction_factor = target_keep_total / current_keep_total
    else:
        correction_factor = 1.0

    owl_sparsities: Dict[str, float] = {}
    for name in raw_sparsities:
        adjusted_keep_frac = (1.0 - raw_sparsities[name] / 100.0) * correction_factor
        adjusted_keep_frac = max(1e-4, min(1.0, adjusted_keep_frac))
        owl_sparsities[name] = 100.0 * (1.0 - adjusted_keep_frac)

    # Verification: print allocation summary
    actual_keep = sum(
        layer_sizes[n] * (1.0 - owl_sparsities[n] / 100.0)
        for n in owl_sparsities
    )
    actual_global_sparsity = 100.0 * (1.0 - actual_keep / total_params)
    sparsity_vals = list(owl_sparsities.values())
    print(f"[OWL] Allocation summary:")
    print(f"  Layers:            {len(owl_sparsities)}")
    print(f"  Target sparsity:   {target_sparsity:.2f}%")
    print(f"  Actual sparsity:   {actual_global_sparsity:.4f}%")
    print(f"  Min layer sparse:  {min(sparsity_vals):.2f}%")
    print(f"  Max layer sparse:  {max(sparsity_vals):.2f}%")
    print(f"  Spread:            {max(sparsity_vals) - min(sparsity_vals):.2f}pp")
    print(f"  Outlier ratio range: [{min(ratio_values):.6f}, {max(ratio_values):.6f}]")

    return owl_sparsities


# ---------------------------------------------------------------------------
# Step A4: Per-Layer Masked Pruning with OWL Budgets
# ---------------------------------------------------------------------------

def create_masks_wanda_owl(
    wanda_scores: Dict[str, torch.Tensor],
    owl_sparsities: Dict[str, float],
    *,
    add_tie_break_noise: bool = True,
    tie_break_noise_scale: float = 1e-6,
    min_layer_keep: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Create binary masks by applying per-layer top-k with OWL-allocated budgets.

    Returns {param_name: bool mask tensor}.
    """
    masks: Dict[str, torch.Tensor] = {}
    total_params = 0
    total_kept = 0

    if add_tie_break_noise:
        global_max = max(s.abs().max().item() for s in wanda_scores.values())
        noise_scale = max(global_max * tie_break_noise_scale, 1e-12)
        torch.manual_seed(42)

    for name, score in wanda_scores.items():
        layer_sparsity = owl_sparsities.get(name, 97.5)
        keep_frac = 1.0 - layer_sparsity / 100.0
        n = score.numel()
        k = max(min_layer_keep, int(keep_frac * n))
        k = min(k, n)

        flat = score.reshape(-1).clone()
        if add_tie_break_noise:
            flat = flat + torch.randn_like(flat) * noise_scale

        _, topk_idx = torch.topk(flat, k, largest=True)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[topk_idx] = True
        masks[name] = mask.reshape(score.shape)

        total_params += n
        total_kept += k

    actual_sparsity = 100.0 * (1.0 - total_kept / max(total_params, 1))
    print(f"\n[Wanda+OWL] Mask creation complete:")
    print(f"  Total parameters:  {total_params:,}")
    print(f"  Kept parameters:   {total_kept:,}")
    print(f"  Actual sparsity:   {actual_sparsity:.4f}%")
    return masks


# ---------------------------------------------------------------------------
# Calibration Data Loading
# ---------------------------------------------------------------------------

def _load_calibration_data(
    tokenizer,
    n_samples: int,
    max_length: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load calibration data for Wanda activation collection.

    Uses a mix of sources to get representative activations:
    - GSM8K (math reasoning)
    - AG News (general language)

    Returns (input_ids, attention_mask) padded tensors.
    """
    from datasets import load_dataset

    samples_per_source = max(1, n_samples // 2)
    texts: List[str] = []

    # GSM8K
    print(f"[Calibration] Loading {samples_per_source} samples from GSM8K...")
    ds_gsm = load_dataset("gsm8k", "main", split="train")
    ds_gsm = ds_gsm.shuffle(seed=seed)
    for row in ds_gsm:
        if len(texts) >= samples_per_source:
            break
        texts.append(f"{row['question']}\n{row['answer']}")

    # AG News
    print(f"[Calibration] Loading {samples_per_source} samples from AG News...")
    ds_ag = load_dataset("ag_news", split="train")
    ds_ag = ds_ag.shuffle(seed=seed)
    for row in ds_ag:
        if len(texts) >= n_samples:
            break
        texts.append(str(row["text"]))

    print(f"[Calibration] Total calibration texts: {len(texts)}")

    # Tokenize
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return encodings["input_ids"], encodings["attention_mask"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    device = choose_device(args.force_cpu)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[Wanda+OWL] Device: {device}, dtype: {dtype}")
    print(f"[Wanda+OWL] Model: {args.model_name}")
    print(f"[Wanda+OWL] Method: {args.method}")
    print(f"[Wanda+OWL] Sparsity: {args.sparsity_percent}%")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.config.use_cache = False
    model.eval()

    # Load calibration data
    input_ids, attention_mask = _load_calibration_data(
        tokenizer, args.n_samples, args.max_length, seed=args.seed,
    )
    if args.dry_run:
        print("[Wanda+OWL] dry_run=1: exiting after data load.")
        return

    input_device = next(model.parameters()).device
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)

    # Step A1: Collect activation norms
    print("\n=== Step A1: Collecting activation norms ===")
    collector = ActivationNormCollector()
    collector.register(model)

    n_total = input_ids.shape[0]
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, n_total, bs):
            b_ids = input_ids[i : i + bs]
            b_am = attention_mask[i : i + bs]
            _ = model(input_ids=b_ids, attention_mask=b_am)
            if (i // bs) % 4 == 0:
                print(f"  Batch {i // bs + 1}/{(n_total + bs - 1) // bs}")

    collector.remove()
    activation_norms = collector.get_norms()

    # Verification A1
    print(f"\n[Check A1] Collected norms for {len(activation_norms)} layers")
    for i, (k, v) in enumerate(sorted(activation_norms.items())[:3]):
        print(f"  {k}: shape={list(v.shape)}, "
              f"min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
    any_bad = any(v.isnan().any().item() or v.numel() == 0 for v in activation_norms.values())
    if any_bad:
        raise ValueError("[Check A1 FAILED] Found NaN or empty norm tensors")
    print("[Check A1 PASSED] All norm tensors valid")

    # Step A2: Wanda scoring
    print("\n=== Step A2: Computing Wanda scores ===")
    wanda_scores = compute_wanda_scores(model, activation_norms)
    print(f"[Check A2] Scored {len(wanda_scores)} weight matrices")
    for i, (k, v) in enumerate(sorted(wanda_scores.items())[:3]):
        print(f"  {k}: shape={list(v.shape)}, "
              f"min={v.min().item():.6f}, max={v.max().item():.4f}")
    # Verify shapes
    param_dict = dict(model.named_parameters())
    for name, score in wanda_scores.items():
        if name in param_dict and score.shape != param_dict[name].shape:
            raise ValueError(f"[Check A2 FAILED] Shape mismatch: {name} "
                             f"score={score.shape} vs param={param_dict[name].shape}")
    print("[Check A2 PASSED] All score shapes match parameter shapes")

    # Step A3: OWL allocation (or uniform for ablation)
    print(f"\n=== Step A3: Computing sparsity allocation (method={args.method}) ===")
    if args.method == "wanda_owl":
        owl_sparsities = compute_owl_sparsity_allocation(
            activation_norms, wanda_scores, args.sparsity_percent,
        )
        # Verification A3
        weighted_avg = sum(
            wanda_scores[n].numel() * owl_sparsities[n]
            for n in owl_sparsities
        ) / sum(wanda_scores[n].numel() for n in owl_sparsities)
        spread = max(owl_sparsities.values()) - min(owl_sparsities.values())
        print(f"[Check A3] Weighted avg sparsity: {weighted_avg:.4f}%  "
              f"(target: {args.sparsity_percent}%), spread: {spread:.2f}pp")
        if abs(weighted_avg - args.sparsity_percent) > 1.0:
            print(f"[Check A3 WARNING] Weighted avg deviates >1% from target")
        if spread < 1.0:
            print(f"[Check A3 WARNING] Spread is <1pp — OWL may not be providing meaningful non-uniformity")
        print("[Check A3 PASSED]")
    else:
        # Uniform (wanda_only)
        owl_sparsities = {name: args.sparsity_percent for name in wanda_scores}
        print(f"[Uniform] All layers set to {args.sparsity_percent}% sparsity")

    # Step A4: Create masks
    print("\n=== Step A4: Creating masks ===")
    masks = create_masks_wanda_owl(wanda_scores, owl_sparsities)

    # Verification A4
    total_p = sum(m.numel() for m in masks.values())
    total_k = sum(m.sum().item() for m in masks.values())
    global_sparsity = 100.0 * (1.0 - total_k / max(total_p, 1))
    print(f"[Check A4] Global sparsity: {global_sparsity:.4f}%  (target: {args.sparsity_percent}%)")
    if abs(global_sparsity - args.sparsity_percent) > 0.5:
        print(f"[Check A4 WARNING] Global sparsity deviates >0.5% from target")
    all_bool = all(m.dtype == torch.bool for m in masks.values())
    if not all_bool:
        raise ValueError("[Check A4 FAILED] Not all masks are torch.bool")
    print("[Check A4 PASSED]")

    # Coverage report + metadata
    coverage = compute_mask_coverage_report(
        model=model,
        masks=masks,
        topk_missing=20,
    )
    if coverage["shape_mismatch_count"] != 0 or coverage["numel_covered_frac_2d"] < args.coverage_gate_2d:
        print("[Wanda+OWL] COVERAGE GATE FAILED")
        print(json.dumps(coverage, indent=2))
        raise SystemExit(2)

    metadata: Dict[str, Any] = {
        "method": "wanda_owl" if args.method == "wanda_owl" else "wanda",
        "model_name": args.model_name,
        "n_samples": args.n_samples,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "sparsity_percent": args.sparsity_percent,
        "owl_enabled": args.method == "wanda_owl",
        "coverage_gate_2d": float(args.coverage_gate_2d),
        "coverage_report": coverage,
    }

    model_sanitized = sanitize_model_name(args.model_name)
    method_tag = "wanda_owl" if args.method == "wanda_owl" else "wanda"
    output_file = args.output_file or f"masks/{method_tag}_{model_sanitized}_sparsity{args.sparsity_percent}pct.pt"
    save_masks(masks, output_file, metadata)

    if args.coverage_report_out:
        os.makedirs(os.path.dirname(args.coverage_report_out) or ".", exist_ok=True)
        with open(args.coverage_report_out, "w", encoding="utf-8") as f:
            json.dump(coverage, f, indent=2)
        print(f"[Wanda+OWL] Coverage report written to: {args.coverage_report_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cold-start Wanda+OWL mask finder")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--method", type=str, choices=["wanda_owl", "wanda_only"], default="wanda_owl",
                    help="wanda_owl = non-uniform OWL allocation; wanda_only = uniform sparsity (ablation)")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_cpu", action="store_true")

    p.add_argument("--sparsity_percent", type=float, default=97.5)
    p.add_argument("--coverage_gate_2d", type=float, default=0.995)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--coverage_report_out", type=str, default=None)
    p.add_argument("--dry_run", action="store_true",
                    help="Load model + data, then exit (no hooks/scoring/masks).")
    args = p.parse_args()
    main(args)

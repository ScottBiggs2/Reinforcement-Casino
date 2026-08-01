#!/usr/bin/env python3
"""
Cold-start CAV v2: "all params suspect" mask generation.

Goal:
- Produce score tensors for essentially all parameters (at least all 2D weights),
  then run the same global selector as warm-start: create_mask_from_scores_gpu_efficient.
- Support both DPO and GRPO calibration modes with identical codepaths; only the
  positive/negative sample definition differs.
- Provide an opt-in rank-collapse protection (weight_abs) by multiplying broadcast
  scores by |W| so score matrices don't collapse to rank-1 under global thresholding.

This script is intended to become the orchestrator default for CAV once validated.
Legacy CAV implementations remain available for rollback/ablations.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is importable (so `import src...` works even without PYTHONPATH).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    pooling_metadata,
    save_masks,
)
from src.utils.mask_coverage_report import compute_mask_coverage_report


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


def compute_contrastive_activation_difference(
    pos_acts: List[torch.Tensor],
    neg_acts: List[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Compute |E[A_pos] - E[A_neg]| to find activation sensitivity.
    """
    if not pos_acts or not neg_acts:
        return None

    X_pos = torch.cat(pos_acts, dim=0).float()
    X_neg = torch.cat(neg_acts, dim=0).float()

    if X_pos.numel() == 0 or X_neg.numel() == 0:
        return None

    mean_pos = X_pos.mean(dim=0)
    mean_neg = X_neg.mean(dim=0)

    diff = (mean_pos - mean_neg).abs()
    return diff

# does this hit attention layers as well? defer to the warm start implementation to check for what is correct
class ActivationCollector:
    """
    Collect pooled activations for many modules in one forward pass using hooks.

    For Linear layers: capture the module input (inp[0]) of shape [B, T, in_features].
    For Embedding layers: capture the module output of shape [B, T, embed_dim].

    We pool across tokens using attention_mask: mean over non-padding tokens.
    """

    def __init__(self, *, include_embeddings: bool):
        self.include_embeddings = include_embeddings
        self._hooks: List[Any] = []
        self._current_attention_mask: Optional[torch.Tensor] = None
        self._store: Dict[str, List[torch.Tensor]] = {}

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        am = self._current_attention_mask
        if am is None or am.dim() != 2 or am.shape[0] != x.shape[0] or am.shape[1] != x.shape[1]:
            return x.mean(dim=1)
        m = am.to(device=x.device).unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        return (x * m).sum(dim=1) / denom

    def register(self, model: nn.Module) -> None:
        # Hook all nn.Linear modules.
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = f"{module_name}.weight"
                self._store.setdefault(key, [])

                def _make_hook(k: str):
                    def _hook(_mod, inp, _out):
                        x = inp[0].detach()
                        if x.dim() != 3:
                            return
                        pooled = self._pool(x).to(dtype=torch.float16).cpu()
                        self._store[k].append(pooled)
                    return _hook

                self._hooks.append(module.register_forward_hook(_make_hook(key)))

        if self.include_embeddings:
            for module_name, module in model.named_modules():
                if isinstance(module, nn.Embedding):
                    key = f"{module_name}.weight"
                    self._store.setdefault(key, [])

                    def _make_hook(k: str):
                        def _hook(_mod, _inp, out):
                            x = out.detach()
                            if x.dim() != 3:
                                return
                            pooled = self._pool(x).to(dtype=torch.float16).cpu()
                            self._store[k].append(pooled)
                        return _hook

                    self._hooks.append(module.register_forward_hook(_make_hook(key)))

        print(f"[ActivationCollector] Hooked groups: {len(self._store)}")

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def collect(
        self,
        *,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._current_attention_mask = response_mask
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        self._current_attention_mask = None

        out: Dict[str, torch.Tensor] = {}
        for k, parts in self._store.items():
            if not parts:
                continue
            out[k] = torch.cat(parts, dim=0)
            self._store[k] = []
        return out

# what is y here, like what does this actually do?
# why arent we going contrast or TCAV - linear probes in this setting seems very odd.
# Whats here right now is a Linear-Probe based method, not a CAV method: 
# Maybe Cursor was going for TCAV style? https://arxiv.org/pdf/1711.11279 
# 

# Clarifying questions: 
# what are positive/negative samples here? The quality of our scores are totally dependent on this choice, so it must be clear and strongly grounded. 
# how are we normalizing activations? There are scaling issues as activations tend to be much larger deeper in LLM layers. 




def build_scores_from_feature_importance(
    model: nn.Module,
    feature_scores: Dict[str, torch.Tensor],
    *,
    weight_abs: bool,
) -> Dict[str, torch.Tensor]:
    """
    Map per-feature scores (1D) into per-parameter score tensors matching each weight shape.

    For 2D weights W[out, in]:
    - If feature_scores matches in_features -> broadcast across out: score = s[None, :]
    - If feature_scores matches out_features -> broadcast across in: score = s[:, None]

    Rank-collapse protection: multiply by |W| when weight_abs=True.
    """
    param_dict = dict(model.named_parameters())
    out: Dict[str, torch.Tensor] = {}

    for pname, s in feature_scores.items():
        if pname not in param_dict:
            continue
        p = param_dict[pname]
        if p.dim() == 2:
            W = p.detach().float().abs().cpu()
            ss = s.reshape(-1).float().cpu()
            if ss.numel() == p.shape[1]:
                base = ss[None, :].expand(p.shape[0], p.shape[1])
            elif ss.numel() == p.shape[0]:
                base = ss[:, None].expand(p.shape[0], p.shape[1])
            else:
                continue
            out[pname] = (base * W) if weight_abs else base
        elif p.dim() == 1:
            # Optional: allow 1D parameters to be scored. If sizes match, use elementwise.
            ss = s.reshape(-1).float().cpu()
            if ss.numel() != p.numel():
                continue
            out[pname] = (ss * p.detach().float().abs().cpu()) if weight_abs else ss

    return out


def _encode_texts_with_response_mask(
    tokenizer, samples: List[Dict[str, str]], max_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_list = []
    response_mask_list = []

    for s in samples:
        prompt_str = s["prompt"]
        full_str = s["prompt"] + "\n" + s["response"]

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        full_ids = tokenizer.encode(full_str, add_special_tokens=True)

        prompt_len = len(prompt_ids)
        full_ids = full_ids[:max_length]
        prompt_len = min(prompt_len, max_length)

        # 0 for prompt, 1 for response
        rmask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        response_mask_list.append(torch.tensor(rmask, dtype=torch.long))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    response_mask = pad_sequence(response_mask_list, batch_first=True, padding_value=0)
    attention_mask = (input_ids != pad_id).long()

    return input_ids, attention_mask, response_mask


def _load_domain_contrast_samples(args) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    from datasets import load_dataset
    pos_samples = []
    neg_samples = []

    # Positive: GSM8K (math reasoning)
    print(f"[Domain Contrast] Loading {args.n_samples} positive samples from GSM8K...")
    ds_pos = load_dataset("gsm8k", "main", split="train")
    if args.seed:
        ds_pos = ds_pos.shuffle(seed=args.seed)

    for row in ds_pos:
        if len(pos_samples) >= args.n_samples: break
        pos_samples.append({
            "prompt": str(row["question"]).strip(),
            "response": str(row["answer"]).strip()
        })

    # Negative: AG News (general language, short text)
    print(f"[Domain Contrast] Loading {args.n_samples} negative samples from AG News...")
    ds_neg = load_dataset("ag_news", split="train")
    if args.seed:
        ds_neg = ds_neg.shuffle(seed=args.seed)

    for row in ds_neg:
        if len(neg_samples) >= args.n_samples: break
        neg_samples.append({
            "prompt": "Please continue the following text:",
            "response": str(row["text"]).strip()
        })

    return pos_samples, neg_samples, "gsm8k_vs_agnews"


def main(args) -> None:
    device = choose_device(args.force_cpu)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

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

    pos_samples, neg_samples, effective_dataset = _load_domain_contrast_samples(args)
    if args.dry_run:
        print("[CAVv2] dry_run=1: exiting after data load.")
        return

    pos_ids, pos_am, pos_rm = _encode_texts_with_response_mask(tokenizer, pos_samples, max_length=args.max_length)
    neg_ids, neg_am, neg_rm = _encode_texts_with_response_mask(tokenizer, neg_samples, max_length=args.max_length)

    input_device = next(model.parameters()).device
    pos_ids, pos_am, pos_rm = pos_ids.to(input_device), pos_am.to(input_device), pos_rm.to(input_device)
    neg_ids, neg_am, neg_rm = neg_ids.to(input_device), neg_am.to(input_device), neg_rm.to(input_device)

    collector = ActivationCollector(include_embeddings=args.include_embeddings)
    collector.register(model)

    def _iter_batches(ids: torch.Tensor, am: torch.Tensor, rm: torch.Tensor, bs: int):
        for i in range(0, ids.shape[0], bs):
            yield ids[i : i + bs], am[i : i + bs], rm[i : i + bs]

    pos_acts: Dict[str, List[torch.Tensor]] = {}
    neg_acts: Dict[str, List[torch.Tensor]] = {}

    print(f"[CAVv2] Collecting activations (pos) n={pos_ids.shape[0]} bs={args.batch_size} ...")
    for b_ids, b_am, b_rm in _iter_batches(pos_ids, pos_am, pos_rm, args.batch_size):
        chunk = collector.collect(model=model, input_ids=b_ids, attention_mask=b_am, response_mask=b_rm)
        for k, v in chunk.items():
            pos_acts.setdefault(k, []).append(v)

    print(f"[CAVv2] Collecting activations (neg) n={neg_ids.shape[0]} bs={args.batch_size} ...")
    for b_ids, b_am, b_rm in _iter_batches(neg_ids, neg_am, neg_rm, args.batch_size):
        chunk = collector.collect(model=model, input_ids=b_ids, attention_mask=b_am, response_mask=b_rm)
        for k, v in chunk.items():
            neg_acts.setdefault(k, []).append(v)

    collector.remove()

    feature_scores: Dict[str, torch.Tensor] = {}
    keys = sorted(set(pos_acts) & set(neg_acts))
    print(f"[CAVv2] Computing contrastive activation difference for {len(keys)} groups ...")
    for k in keys:
        diff = compute_contrastive_activation_difference(pos_acts[k], neg_acts[k])
        if diff is None:
            continue
        feature_scores[k] = diff

    # Map to per-parameter score tensors.
    scores_dict = build_scores_from_feature_importance(model, feature_scores, weight_abs=args.weight_abs)
    if not scores_dict:
        raise ValueError("CAVv2 produced an empty scores_dict; cannot create masks.")

    masks = create_mask_from_scores_gpu_efficient(
        scores_dict,
        args.sparsity_percent,
        device=device,
        add_tie_break_noise=True,
        min_layer_keep_ratio=args.min_layer_keep_ratio,
        local_pool=args.local_pool,
    )

    # Coverage report + metadata.
    coverage = compute_mask_coverage_report(
        model=model,
        masks=masks,
        topk_missing=20,
    )
    if coverage["shape_mismatch_count"] != 0 or coverage["numel_covered_frac_2d"] < args.coverage_gate_2d:
        print("[CAVv2] COVERAGE GATE FAILED")
        print(json.dumps(coverage, indent=2))
        raise SystemExit(2)

    metadata: Dict[str, Any] = {
        "method": "cav",
        "cav_impl": "cold_v2_all_params",
        "mode": args.mode,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "effective_dataset": effective_dataset,
        "n_samples": args.n_samples,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "sparsity_percent": args.sparsity_percent,
        "include_embeddings": bool(args.include_embeddings),
        "weight_abs": bool(args.weight_abs),
        "coverage_gate_2d": float(args.coverage_gate_2d),
        "coverage_report": coverage,
        **pooling_metadata(local_pool=args.local_pool, min_layer_keep_ratio=args.min_layer_keep_ratio),
    }

    model_sanitized = sanitize_model_name(args.model_name)
    output_file = args.output_file or f"masks/cold_cav_v2_{args.mode}_{model_sanitized}_sparsity{args.sparsity_percent}pct.pt"
    save_masks(masks, output_file, metadata)

    if args.coverage_report_out:
        os.makedirs(os.path.dirname(args.coverage_report_out) or ".", exist_ok=True)
        with open(args.coverage_report_out, "w", encoding="utf-8") as f:
            json.dump(coverage, f, indent=2)
        print(f"[CAVv2] Coverage report written to: {args.coverage_report_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cold-start CAV v2 (all params suspect) mask finder")
    p.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    p.add_argument("--mode", type=str, choices=["dpo", "grpo"], default="dpo")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_cpu", action="store_true")

    p.add_argument("--sparsity_percent", type=float, default=97.5)
    p.add_argument(
        "--min_layer_keep_ratio",
        type=float,
        default=DEFAULT_MIN_LAYER_KEEP_RATIO,
        help="Per-tensor keep floor for global masking (0.0 for pure global).",
    )
    p.add_argument("--local_pool", action="store_true")

    p.add_argument("--include_embeddings", action="store_true", help="Also score embedding matrices (default off).")
    p.add_argument(
        "--weight_abs",
        action="store_true",
        default=True,
        help="Rank-collapse protection (True by default): multiply broadcast scores by |W| to map Activation difference to Weight score.",
    )
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--coverage_gate_2d", type=float, default=0.995)
    p.add_argument("--coverage_report_out", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="Load model + data, then exit (no hooks/probes/masks).")
    args = p.parse_args()
    main(args)


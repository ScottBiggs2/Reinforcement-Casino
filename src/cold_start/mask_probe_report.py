#!/usr/bin/env python3
"""Linear-probe (CAV-style) report for a single sparse mask: chosen vs rejected activations on MLP hooks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.cold_start.inference_mask_finder import load_calibration_samples
from src.cold_start.mask_to_cka import apply_mask, load_masks, restore_weights
from src.cold_start.utils.activation_hooks import FeatureExtractor, infer_model_input_device
from src.cold_start.utils.cav_probes import CAVProbeScorer


def _summarize_probe_report(report: Dict[str, Any]) -> Dict[str, Any]:
    layers = report.get("layers") or {}
    train_accs: List[float] = []
    cv_accs: List[float] = []
    for _k, row in layers.items():
        if isinstance(row, dict):
            ta = row.get("train_accuracy")
            if ta is not None:
                train_accs.append(float(ta))
            cv = row.get("cv_accuracy_mean")
            if cv is not None:
                cv_accs.append(float(cv))
    return {
        "n_probe_layers": len(layers),
        "mean_train_accuracy": sum(train_accs) / len(train_accs) if train_accs else None,
        "mean_cv_accuracy_mean": sum(cv_accs) / len(cv_accs) if cv_accs else None,
    }


def run_probe_report(
    mask_path: str,
    model_name: str,
    out_json: str,
    *,
    mode: str = "grpo",
    dataset_name: Optional[str] = None,
    device: str = "cuda",
    n_samples: int = 64,
    batch_size: int = 4,
    max_length: int = 512,
    seed: int = 42,
) -> Dict[str, Any]:
    chosen_texts, rejected_texts = load_calibration_samples(
        n_samples=n_samples,
        seed=seed,
        mode=mode,
        dataset_name=dataset_name,
    )
    if len(chosen_texts) < 4 or len(rejected_texts) < 4:
        raise RuntimeError(
            f"Need >=4 calibration strings; got chosen={len(chosen_texts)} rejected={len(rejected_texts)}"
        )

    masks, mask_meta = load_masks(mask_path)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if str(torch_device) == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")

    print(f"Loading model {model_name!r} on {torch_device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if str(torch_device) == "cuda" else None,
    )
    if str(torch_device) == "cpu":
        model.to(torch_device)
    model.config.use_cache = False
    model.eval()
    input_device = infer_model_input_device(model)

    extractor = FeatureExtractor()
    extractor.register(model)
    snap = apply_mask(model, masks)
    try:
        print("[probe] Collecting chosen activations...")
        pos_acts = extractor.collect(
            model,
            tokenizer,
            chosen_texts,
            input_device,
            max_length=max_length,
            batch_size=batch_size,
        )
        print("[probe] Collecting rejected / contrast activations...")
        neg_acts = extractor.collect(
            model,
            tokenizer,
            rejected_texts,
            input_device,
            max_length=max_length,
            batch_size=batch_size,
        )
    finally:
        extractor.remove()
        restore_weights(model, snap)

    scorer = CAVProbeScorer()
    _scores, probe_inner = scorer.score_with_probe_report(pos_acts, neg_acts, mag_weight=1.0)
    summary = _summarize_probe_report(probe_inner)
    out: Dict[str, Any] = {
        "mask_path": os.path.abspath(mask_path),
        "model_name": model_name,
        "calibration_mode": mode,
        "calibration_dataset": dataset_name,
        "n_samples_requested": n_samples,
        "mask_metadata_excerpt": mask_meta if isinstance(mask_meta, dict) else None,
        "summary": summary,
        "probe_report": probe_inner,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[probe] Wrote {out_json}")
    print(f"[probe] summary: {summary}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Per-mask linear probe report (GRPO/DPO calibration).")
    p.add_argument("mask", type=str, help="Mask .pt path")
    p.add_argument("--output", "-o", type=str, required=True, help="Output JSON path")
    p.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id (must match mask parameter names).",
    )
    p.add_argument("--mode", type=str, default="grpo", choices=["grpo", "dpo"])
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HF dataset id or override; default follows mode (OpenR1-Math-220k for grpo).",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not os.path.isfile(args.mask):
        print(f"Mask not found: {args.mask}", file=sys.stderr)
        sys.exit(1)

    run_probe_report(
        args.mask,
        args.model_name,
        args.output,
        mode=args.mode,
        dataset_name=args.dataset_name,
        device=args.device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

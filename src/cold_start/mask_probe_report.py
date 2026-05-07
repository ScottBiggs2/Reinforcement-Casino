#!/usr/bin/env python3
"""Linear-probe (CAV-style) report for a single sparse mask: chosen vs rejected activations on MLP hooks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.cold_start.inference_mask_finder import (
    DPO_DATASET_NAME,
    GRPO_DATASET_NAME,
    load_calibration_samples,
)
from src.cold_start.mask_to_cka import apply_mask, load_masks, restore_weights
from src.cold_start.probe_builtin_datasets import (
    PROBE_DATASETS,
    build_concatenated_texts_and_slices,
    layer_index_from_hook_name,
    summarize_layer_scores,
    train_linear_probes_cv,
    validate_probe_datasets,
)
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


def _filter_activation_layers(
    acts: Dict[str, torch.Tensor], layer_stride: int
) -> Dict[str, torch.Tensor]:
    if layer_stride <= 1:
        return acts
    keys = list(acts.keys())
    if not keys:
        return acts
    max_li = max(layer_index_from_hook_name(k) for k in keys)
    keep_idx = set(range(0, max_li + 1, layer_stride))
    keep_idx.add(max_li)
    return {k: v for k, v in acts.items() if layer_index_from_hook_name(k) in keep_idx}


def _builtin_linear_probes(
    model,
    tokenizer,
    extractor: FeatureExtractor,
    input_device: torch.device,
    *,
    property_names: Sequence[str],
    batch_size: int,
    max_length: int,
    cv_folds: int,
    layer_stride: int,
) -> Dict[str, Any]:
    """Builtin probe properties (syntax, semantics, …) on the **current** model weights."""
    validate_probe_datasets()
    all_texts, prop_slices, labels_by_prop = build_concatenated_texts_and_slices(property_names)
    print(f"[probe-builtin] One forward over {len(all_texts)} texts across {len(property_names)} properties...")
    full_acts = extractor.collect(
        model,
        tokenizer,
        all_texts,
        input_device,
        max_length=max_length,
        batch_size=batch_size,
    )
    out: Dict[str, Any] = {}
    for prop in property_names:
        slc = prop_slices[prop]
        sliced = {name: acts[slc] for name, acts in full_acts.items()}
        sliced = _filter_activation_layers(sliced, layer_stride)
        labels = labels_by_prop[prop]
        scores, diag = train_linear_probes_cv(sliced, labels, cv=cv_folds)
        out[prop] = {
            "description": PROBE_DATASETS[prop].get("description"),
            "summary": summarize_layer_scores(scores),
            "per_layer_cv_accuracy": scores,
            "diagnostics": diag,
        }
    return out


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
    builtin_properties: Optional[Sequence[str]] = None,
    builtin_cv_folds: int = 3,
    builtin_layer_stride: int = 1,
) -> Dict[str, Any]:
    chosen_texts, rejected_texts = load_calibration_samples(
        n_samples=n_samples,
        seed=seed,
        mode=mode,
        dataset_name=dataset_name,
    )
    effective_cal_dataset = dataset_name or (
        DPO_DATASET_NAME if mode == "dpo" else GRPO_DATASET_NAME
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
    builtin_by_property: Optional[Dict[str, Any]] = None
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

        if builtin_properties:
            names = [x.strip() for x in builtin_properties if x and str(x).strip()]
            if names:
                builtin_by_property = _builtin_linear_probes(
                    model,
                    tokenizer,
                    extractor,
                    input_device,
                    property_names=names,
                    batch_size=batch_size,
                    max_length=max_length,
                    cv_folds=builtin_cv_folds,
                    layer_stride=builtin_layer_stride,
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
        "calibration_dataset_effective": effective_cal_dataset,
        "n_samples_requested": n_samples,
        "mask_metadata_excerpt": mask_meta if isinstance(mask_meta, dict) else None,
        "summary": summary,
        "probe_report": probe_inner,
        "breakdown_by_dataset": {
            "calibration": {
                "kind": "preference_contrast" if mode == "dpo" else "grpo_chosen_vs_prompt_only",
                "dataset": effective_cal_dataset,
                "summary": summary,
                "per_layer": probe_inner.get("layers"),
            },
        },
    }
    if builtin_by_property:
        for prop, block in builtin_by_property.items():
            out["breakdown_by_dataset"][f"builtin_{prop}"] = {
                "kind": "builtin_binary_probe",
                "property": prop,
                "description": block.get("description"),
                "summary": block.get("summary"),
                "per_layer_cv_accuracy": block.get("per_layer_cv_accuracy"),
                "diagnostics": block.get("diagnostics"),
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
    p.add_argument(
        "--probe-builtin-datasets",
        type=str,
        default="all",
        help="Builtin probe corpora: 'all' | 'none' | comma keys (syntax,semantics,history,geography,math).",
    )
    p.add_argument("--probe-builtin-cv-folds", type=int, default=3)
    p.add_argument("--probe-builtin-layer-stride", type=int, default=1)
    args = p.parse_args()

    if not os.path.isfile(args.mask):
        print(f"Mask not found: {args.mask}", file=sys.stderr)
        sys.exit(1)

    raw_builtin = (args.probe_builtin_datasets or "none").strip().lower()
    if raw_builtin in ("", "none", "off", "0"):
        builtin_props = None
    elif raw_builtin == "all":
        builtin_props = list(PROBE_DATASETS.keys())
    else:
        builtin_props = [s.strip() for s in raw_builtin.split(",") if s.strip()]

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
        builtin_properties=builtin_props,
        builtin_cv_folds=args.probe_builtin_cv_folds,
        builtin_layer_stride=args.probe_builtin_layer_stride,
    )


if __name__ == "__main__":
    main()

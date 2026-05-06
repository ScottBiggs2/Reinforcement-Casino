#!/usr/bin/env python3
"""Dense-trained builtin probes evaluated on dense vs masked subnetworks.

This script is intentionally Irene-style but adapted for the mask suite:

* Train one L2 logistic probe per MLP layer on **dense** activations only.
* For each builtin probe property (syntax/semantics/factual/math):
  - Use a fixed train/test split of the builtin texts.
  - Train probes on dense-train activations.
  - Evaluate on dense-test and masked-test activations for every mask.
* Write a JSON summary and optional PNGs that compare dense vs each mask.

The mask interpretation suite invokes this as a sub-process and merges a
lightweight index into ``suite_summary.json`` so downstream tooling can
find the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cold_start.mask_to_cka import apply_mask, load_masks, restore_weights  # reuse mask utils
from src.cold_start.probe_builtin_datasets import (
    PROBE_DATASETS,
    build_concatenated_texts_and_slices,
    layer_index_from_hook_name,
    validate_probe_datasets,
)
from src.cold_start.utils.activation_hooks import FeatureExtractor, infer_model_input_device


@dataclass
class ProbeScores:
    dense_train: float
    dense_test: float
    masked_test: float


def _collect_builtin_activations(
    model,
    tokenizer,
    extractor: FeatureExtractor,
    input_device: torch.device,
    *,
    properties: Sequence[str],
    batch_size: int,
    max_length: int,
) -> Tuple[
    List[str],
    Dict[str, torch.Tensor],
    List[str],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    Dict[str, slice],
    Dict[str, np.ndarray],
]:
    """Run one forward over all builtin probe texts and prepare splits.

    Returns:
        full_acts: {layer_name: [N_texts, dim]} activations for the dense model.
        prop_names: ordered list of property keys actually used.
        splits: {prop: (train_idx, test_idx)} index arrays into the *global* text list.
    """
    validate_probe_datasets()
    all_texts, prop_slices, labels_by_prop = build_concatenated_texts_and_slices(properties)
    print(
        f"[dense-vs-mask] builtin probes: {len(properties)} properties, "
        f"{len(all_texts)} texts total"
    )
    full_acts = extractor.collect(
        model,
        tokenizer,
        all_texts,
        input_device,
        max_length=max_length,
        batch_size=batch_size,
    )

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    used_props: List[str] = []
    for prop in properties:
        slc = prop_slices[prop]
        labels = labels_by_prop[prop]
        # Indices are **local** to this property slice (0..n_prop-1) so they stay valid
        # after we slice activations down to this property.
        idx = np.arange(0, slc.stop - slc.start, dtype=np.int64)
        # 70/30 stratified split for a small held-out test set.
        try:
            tr, te = train_test_split(
                idx,
                test_size=0.3,
                random_state=42,
                stratify=labels,
            )
        except ValueError:
            # Fallback: simple contiguous split if stratification fails.
            if len(idx) < 4:
                continue
            cut = int(round(len(idx) * 0.7))
            tr, te = idx[:cut], idx[cut:]
        splits[prop] = (tr, te)
        used_props.append(prop)
    return all_texts, full_acts, used_props, splits, prop_slices, labels_by_prop


def _train_and_eval_layer_probe(
    dense_acts: torch.Tensor,
    masked_acts: torch.Tensor,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Optional[ProbeScores]:
    """Train a single linear probe on dense activations and score dense/masked test.

    All inputs are for a single layer; shapes:
        dense_acts: [N, D]
        masked_acts: [N, D]
        labels: [N]
    """
    if dense_acts.shape != masked_acts.shape:
        return None
    n = dense_acts.shape[0]
    if n < 4 or len(np.unique(labels)) < 2:
        return None

    tr = np.asarray(train_idx, dtype=np.int64)
    te = np.asarray(test_idx, dtype=np.int64)
    if tr.size < 2 or te.size < 2:
        return None

    X_train = dense_acts[tr].float().numpy()
    y_train = labels[tr]
    X_test_dense = dense_acts[te].float().numpy()
    X_test_mask = masked_acts[te].float().numpy()
    y_test = labels[te]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_dense_sc = scaler.transform(X_test_dense)
    X_test_mask_sc = scaler.transform(X_test_mask)

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        max_iter=500,
        random_state=42,
    )
    clf.fit(X_train_sc, y_train)

    train_acc = float((clf.predict(X_train_sc) == y_train).mean())
    dense_acc = float((clf.predict(X_test_dense_sc) == y_test).mean())
    mask_acc = float((clf.predict(X_test_mask_sc) == y_test).mean())
    return ProbeScores(train_acc, dense_acc, mask_acc)


def _dense_vs_mask_for_property(
    prop: str,
    *,
    dense_acts: Mapping[str, torch.Tensor],
    masked_acts_by_mask: Mapping[str, Mapping[str, torch.Tensor]],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-layer scores for one property across all masks."""
    layer_names = sorted(dense_acts.keys(), key=layer_index_from_hook_name)
    per_layer: Dict[str, Dict[str, Any]] = {}

    for lname in layer_names:
        da = dense_acts[lname]
        if da.shape[0] != labels.shape[0]:
            continue
        row: Dict[str, Any] = {}
        # Evaluate once for each mask, but reuse the same dense-trained classifier.
        for mask_label, acts_map in masked_acts_by_mask.items():
            ma = acts_map.get(lname)
            if ma is None:
                continue
            scores = _train_and_eval_layer_probe(da, ma, labels, train_idx, test_idx)
            if scores is None:
                continue
            # We overwrite dense_* if multiple masks share a layer name, but they should
            # all be identical because training only depends on dense activations.
            row.setdefault("dense_train_accuracy", scores.dense_train)
            row.setdefault("dense_test_accuracy", scores.dense_test)
            row.setdefault("masks", {})
            row["masks"][mask_label] = {
                "masked_test_accuracy": scores.masked_test,
                "delta_vs_dense": scores.masked_test - scores.dense_test,
            }
        if row:
            per_layer[lname] = row

    # Summary: average over layers, then over masks.
    summary_masks: Dict[str, Dict[str, float]] = {}
    for lname, row in per_layer.items():
        dense_test = row.get("dense_test_accuracy")
        masks = row.get("masks") or {}
        for mlabel, mrow in masks.items():
            acc = mrow.get("masked_test_accuracy")
            if dense_test is None or acc is None:
                continue
            delta = acc - dense_test
            bucket = summary_masks.setdefault(
                mlabel,
                {"masked_mean": 0.0, "dense_mean": 0.0, "delta_mean": 0.0, "n_layers": 0},
            )
            bucket["masked_mean"] += acc
            bucket["dense_mean"] += dense_test
            bucket["delta_mean"] += delta
            bucket["n_layers"] += 1
    for mlabel, bucket in summary_masks.items():
        n = max(1, bucket["n_layers"])
        bucket["masked_mean"] /= n
        bucket["dense_mean"] /= n
        bucket["delta_mean"] /= n

    dense_mean = None
    if per_layer:
        vals = [row.get("dense_test_accuracy") for row in per_layer.values() if row.get("dense_test_accuracy") is not None]
        if vals:
            dense_mean = float(sum(vals) / len(vals))

    return {
        "property": prop,
        "description": PROBE_DATASETS[prop].get("description"),
        "dense_test_mean_accuracy": dense_mean,
        "per_mask_summary": summary_masks,
        "per_layer": per_layer,
    }


def run_dense_vs_mask_probes(
    checkpoint_dir: str,
    model_name: str,
    mask_paths: Sequence[str],
    mask_labels: Sequence[str],
    out_json: str,
    *,
    builtin_properties: Sequence[str],
    batch_size: int = 4,
    max_length: int = 512,
) -> Dict[str, Any]:
    """Main engine: dense-trained builtin probes evaluated on dense + masked."""
    if len(mask_paths) != len(mask_labels):
        raise ValueError("mask_paths and mask_labels lengths must match")

    print(f"[dense-vs-mask] checkpoint_dir={checkpoint_dir}")
    print(f"[dense-vs-mask] model_name={model_name}")
    print(f"[dense-vs-mask] masks={len(mask_paths)}  properties={list(builtin_properties)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model.config.use_cache = False
    model.eval()
    input_device = infer_model_input_device(model)

    extractor = FeatureExtractor()
    extractor.register(model)
    try:
        all_texts, dense_acts, props, splits, prop_slices, labels_by_prop = _collect_builtin_activations(
            model,
            tokenizer,
            extractor,
            input_device,
            properties=builtin_properties,
            batch_size=batch_size,
            max_length=max_length,
        )

        # Pre-collect masked activations for all masks (one forward per mask).
        masked_acts_by_mask: Dict[str, Dict[str, torch.Tensor]] = {}
        for mpath, mlabel in zip(mask_paths, mask_labels):
            print(f"[dense-vs-mask] collecting masked activations for {mlabel} ← {mpath}")
            masks, _meta = load_masks(mpath)
            snap = apply_mask(model, masks)
            try:
                acts = extractor.collect(
                    model,
                    tokenizer,
                    all_texts,
                    input_device,
                    max_length=max_length,
                    batch_size=batch_size,
                )
            finally:
                restore_weights(model, snap)
            # The FeatureExtractor already uses hook names and aligns by text index, so we
            # simply trust that ``acts`` has the same [N_texts, D] shapes as ``dense_acts``.
            masked_acts_by_mask[mlabel] = acts

        # Build per-property reports.
        per_property: List[Dict[str, Any]] = []
        for prop in props:
            slc = prop_slices[prop]
            labels = labels_by_prop[prop]
            tr, te = splits[prop]
            # Slice activations for just this property's texts.
            dense_prop_acts = {
                lname: acts[slc] for lname, acts in dense_acts.items()
            }
            masked_prop_acts = {
                mlabel: {
                    lname: acts[slc] for lname, acts in masked_acts_by_mask[mlabel].items()
                }
                for mlabel in masked_acts_by_mask
            }
            report = _dense_vs_mask_for_property(
                prop,
                dense_acts=dense_prop_acts,
                masked_acts_by_mask=masked_prop_acts,
                labels=labels,
                train_idx=tr,
                test_idx=te,
            )
            per_property.append(report)
    finally:
        extractor.remove()

    out: Dict[str, Any] = {
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "model_name": model_name,
        "mask_labels": list(mask_labels),
        "mask_paths": [os.path.abspath(p) for p in mask_paths],
        "properties": per_property,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[dense-vs-mask] wrote {out_json}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dense-trained builtin probes evaluated on dense vs masked subnetworks."
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="HF-style checkpoint directory for the dense reference model.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id (used for tokenizer and config).",
    )
    p.add_argument(
        "--masks",
        nargs="+",
        required=True,
        help="Mask .pt files (same architecture as checkpoint).",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Short labels for each mask (same length as --masks).",
    )
    p.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output JSON path for dense-vs-mask probe report.",
    )
    p.add_argument(
        "--builtin-datasets",
        type=str,
        default="all",
        help="Builtin Irene corpora: 'all'|'none'|comma keys (syntax,semantics,math,...)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args()

    raw = (args.builtin_datasets or "none").strip().lower()
    if raw in ("", "none", "off", "0"):
        print("[dense-vs-mask] builtin-datasets=none; nothing to do.", file=sys.stderr)
        sys.exit(0)
    elif raw == "all":
        props = list(PROBE_DATASETS.keys())
    else:
        props = [s.strip() for s in raw.split(",") if s.strip()]

    run_dense_vs_mask_probes(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        mask_paths=args.masks,
        mask_labels=args.labels,
        out_json=args.output_json,
        builtin_properties=props,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()


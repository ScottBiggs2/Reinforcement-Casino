#!/usr/bin/env python3
"""
Probe pairwise-ranking comparison across fully-trained model checkpoints.

This is the "scheme D" pipeline: instead of loading one base model and
applying a weight mask on the fly (probe_pair_masks.py), we load each
fully-trained checkpoint separately. That's what you need when the
research question is "dense-trained vs sparse-trained representations"
— i.e. compare a DPO/GRPO run that updated all weights against one that
only updated weights inside a mask.

Input: JSON list of {label, path} where path is either a HF hub model ID
or a local checkpoint directory containing config.json + model-*.safetensors.

Each checkpoint is loaded in turn, its mean-pooled MLP activations are
collected over the same text batch, and the Bradley-Terry pairwise probe
is trained per layer, per property. Results land in one JSON plus
absolute and Δ heatmaps.

Properties (6 by default):
    syntax / semantics / factual / math   (generic benchmarks)
    dpo_preference                         (tulu3 chosen vs rejected)
    grpo_preference                        (OpenR1-Math correct vs incorrect)

Usage:
    python src/analysis/probe_checkpoints.py \\
        --ckpts_json ckpts_scheme_d.json \\
        --output_dir /scratch/$USER/probe_checkpoints/
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))
from cold_start.utils.activation_hooks import FeatureExtractor
from analysis.probe_analysis import (
    _layer_index,
    _load_hf_probe_datasets,
    apply_mask,
    load_mask,
    no_mask,
    plot_delta_heatmap,
    plot_probe_heatmap,
)
from analysis.probe_analysis_pair import train_pairwise_probes


DEFAULT_BASELINE_HF_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BASELINE_LABEL = "Baseline (Instruct)"


def parse_args():
    p = argparse.ArgumentParser(
        description="Pairwise ranking probe over fully-trained checkpoints"
    )
    p.add_argument("--ckpts_json", default=None,
                   help="JSON list of {label, path} describing checkpoints "
                        "to probe. Example entries include both HF IDs and "
                        "local directories containing a HF checkpoint.")
    p.add_argument("--baseline_path", default=DEFAULT_BASELINE_HF_ID,
                   help="Untrained reference model (HF ID or local path). "
                        "Used when --skip_baseline is NOT set. Default: "
                        "meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--baseline_label", default=DEFAULT_BASELINE_LABEL)
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip the untrained baseline (disables Δ heatmap)")
    p.add_argument("--output_dir", default="probe_checkpoints")

    # Probe / probe data options (mirror probe_pair_masks.py)
    p.add_argument("--layer_stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--pairs_per_pos", type=int, default=2)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--holdout_frac", type=float, default=0.4,
                   help="Stratified per-class fraction reserved as untouched "
                        "holdout before CV (default: 0.4 = 60/40 train/test)")
    p.add_argument("--use_holdout_as_test", action="store_true", default=True,
                   help="Report holdout accuracy as primary 'test' metric "
                        "(default: on). CV diagnostics remain in cv_* fields.")
    p.add_argument("--no_use_holdout_as_test", dest="use_holdout_as_test",
                   action="store_false")
    p.add_argument("--probe_C", type=float, default=0.1,
                   help="L2 regularization strength for the logistic probe "
                        "(default: 0.1, stronger than sklearn default of 1.0 "
                        "to curb overfitting in p>>n regime).")
    p.add_argument("--samples_per_class", type=int, default=None,
                   help="Per-class cap for the 4 benchmark probes")
    p.add_argument("--preference_samples_per_class", type=int, default=1500,
                   help="Per-class cap for tulu3/open-r1 preference probes")
    p.add_argument("--no_preference", action="store_true")
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--probe_cache", default=None)

    p.add_argument("--skip_missing", action="store_true",
                   help="If a checkpoint dir is missing, warn and skip "
                        "instead of failing")
    return p.parse_args()


def load_probe_datasets(args):
    """Same cache/balance logic as probe_pair_masks.py. Uses v3 cache layout
    that supports unequal per-property sizes (preference probes have more
    samples than the 4 benchmarks)."""
    cache_dir = args.dataset_cache_dir or os.environ.get("HF_DATASETS_CACHE")
    _CACHE_VERSION = "v3"
    if args.probe_cache:
        probe_cache_path = args.probe_cache
    else:
        n_tag = args.samples_per_class if args.samples_per_class is not None else "all"
        pref_tag = "none" if args.no_preference else f"p{args.preference_samples_per_class}"
        cache_name = (
            f"probe_dataset_cache_{_CACHE_VERSION}_n{n_tag}_{pref_tag}_seed42.json"
        )
        probe_cache_path = os.path.join(args.output_dir, os.pardir, cache_name)
        probe_cache_path = os.path.normpath(probe_cache_path)

    if os.path.exists(probe_cache_path):
        print(f"[data] Loading cached probe dataset from {probe_cache_path}")
        with open(probe_cache_path) as f:
            datasets = json.load(f)
        for prop in datasets:
            datasets[prop]["examples"] = [
                (t, l) for t, l in datasets[prop]["examples"]
            ]
    else:
        datasets = _load_hf_probe_datasets(
            samples_per_class=args.samples_per_class,
            preference_samples_per_class=args.preference_samples_per_class,
            cache_dir=cache_dir,
            include_preference=not args.no_preference,
        )
        os.makedirs(os.path.dirname(probe_cache_path), exist_ok=True)
        with open(probe_cache_path, "w") as f:
            json.dump(datasets, f)
        print(f"[data] Cached probe dataset → {probe_cache_path}")

    # Balanced pos/neg per-property, but allow different N across properties.
    for prop_name, prop_data in datasets.items():
        n = len(prop_data["examples"])
        lbls = [l for _, l in prop_data["examples"]]
        n_pos = sum(1 for l in lbls if l == 1)
        n_neg = sum(1 for l in lbls if l == 0)
        assert n_pos == n_neg, (
            f"'{prop_name}' unbalanced: {n_pos} pos vs {n_neg} neg (n={n})"
        )
        print(f"[data]   {prop_name:18s}  n={n}  (pos={n_pos}, neg={n_neg})")
    return datasets


def evaluate_checkpoint(ckpt_path, label, all_texts, args,
                         property_slices, probe_datasets,
                         sample_indices_ref, mask_path=None,
                         preloaded=None):
    """Load one checkpoint, collect activations, train per-property probes.

    Args:
        mask_path: Optional HF-format or .pt mask. When set, apply_mask zeroes
            out the mask=0 positions before the forward pass so the probe sees
            only the sub-network formed by mask=1 weights. The original
            weights are restored after collection so callers can reuse the
            loaded model for a different mask (see `preloaded`).
        preloaded: Optional tuple (model, tokenizer) to reuse across probe
            runs. If provided we skip from_pretrained; the caller owns the
            lifetime and should `del`/empty_cache when done.

    sample_indices_ref is resolved lazily on the first call (all checkpoints
    are assumed to share the same architecture — number of MLP layers).
    Subsequent calls pass in the resolved set to skip re-counting.
    """
    print(f"\n[main] ===== {label} =====")
    print(f"       path: {ckpt_path}")
    if mask_path:
        print(f"       mask: {mask_path}")
    t0 = time.time()

    if preloaded is not None:
        model, tokenizer = preloaded
        print(f"  [{label}] reusing preloaded model")
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        print(f"  [{label}] model+tokenizer load: {time.time() - t0:.1f}s")

    # Resolve sampling indices on first checkpoint (assumes all share arch).
    if sample_indices_ref["resolved"] is None:
        n_mlp = sum(
            1 for name, _ in model.named_modules()
            if name.endswith("down_proj")
        )
        sample_indices = list(range(0, n_mlp, args.layer_stride))
        if (n_mlp - 1) not in sample_indices:
            sample_indices.append(n_mlp - 1)
        sample_indices_ref["resolved"] = sample_indices
        sample_indices_ref["n_mlp"] = n_mlp
        print(f"  [{label}] n_mlp_layers={n_mlp}, "
              f"sampling {len(sample_indices)} layers: {sample_indices}")
    sample_indices = sample_indices_ref["resolved"]
    sample_index_set = set(sample_indices)

    extractor = FeatureExtractor().register(model)
    device = next(model.parameters()).device

    # Apply mask for sub-network probing. Context manager restores weights
    # on exit, so the caller can reuse the model for a different mask.
    if mask_path:
        t_mask = time.time()
        mask_dict = load_mask(mask_path)
        mask_ctx = apply_mask(model, mask_dict)
        print(f"  [{label}] mask load: {time.time() - t_mask:.1f}s")
    else:
        mask_ctx = no_mask()

    t0 = time.time()
    with mask_ctx:
        activations = extractor.collect(
            model, tokenizer, all_texts, device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    print(f"  [{label}] activation pass over {len(all_texts)} texts: "
          f"{time.time() - t0:.1f}s")

    sampled_acts = {
        name: acts for name, acts in activations.items()
        if _layer_index(name) in sample_index_set
    }

    prop_results = {}
    cfg_t0 = time.time()
    cfg_notconv = cfg_degen = cfg_nonfinite = 0
    cfg_reason_counts = {}

    for prop_name, prop_data in probe_datasets.items():
        slc = property_slices[prop_name]
        labels_arr = np.array([lbl for _, lbl in prop_data["examples"]])
        prop_acts = {name: acts[slc] for name, acts in sampled_acts.items()}

        prop_t0 = time.time()
        layer_accs = train_pairwise_probes(
            prop_acts, labels_arr,
            cv=args.cv_folds,
            pairs_per_pos=args.pairs_per_pos,
            n_jobs=args.n_jobs,
            holdout_frac=args.holdout_frac,
            use_holdout_as_test=args.use_holdout_as_test,
            probe_C=args.probe_C,
        )
        prop_results[prop_name] = layer_accs
        prop_dt = time.time() - prop_t0

        test_vals = [v["test"] for v in layer_accs.values()]
        train_vals = [v["train"] for v in layer_accs.values()]
        holdout_vals = [
            v.get("holdout_test") for v in layer_accs.values()
            if v.get("holdout_test") is not None
        ]
        n_notconv = sum(1 for v in layer_accs.values() if not v["converged"])
        n_degen = sum(1 for v in layer_accs.values() if v["degenerate"])
        nonfinite_total = int(sum(v.get("nonfinite_count", 0) for v in layer_accs.values()))
        reasons = {}
        for v in layer_accs.values():
            r = v.get("degenerate_reason")
            if r:
                reasons[r] = reasons.get(r, 0) + 1

        cfg_notconv += n_notconv
        cfg_degen += n_degen
        cfg_nonfinite += nonfinite_total
        for k, v in reasons.items():
            cfg_reason_counts[k] = cfg_reason_counts.get(k, 0) + v

        reason_msg = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items())) or "none"
        metric_msg = (
            f"  {prop_name:18s} ({prop_dt:5.1f}s): "
            f"test={np.nanmean(test_vals):.3f}  "
            f"train={np.nanmean(train_vals):.3f}  "
            f"gap={np.nanmean(train_vals) - np.nanmean(test_vals):+.3f}  "
        )
        if holdout_vals:
            metric_msg += f"holdout={np.nanmean(holdout_vals):.3f}  "
        metric_msg += (
            f"non_conv={n_notconv}/{len(layer_accs)}  "
            f"degen={n_degen}/{len(layer_accs)}  "
            f"nonfinite={nonfinite_total}  "
            f"reasons={reason_msg}"
        )
        print(metric_msg, flush=True)

    cfg_reason_msg = ", ".join(
        f"{k}:{v}" for k, v in sorted(cfg_reason_counts.items())
    ) or "none"
    print(
        f"  [{label}] diagnostics: non_conv={cfg_notconv} degen={cfg_degen} "
        f"nonfinite={cfg_nonfinite} reasons={cfg_reason_msg}",
        flush=True,
    )
    print(f"  [{label}] total probe time: {time.time() - cfg_t0:.1f}s",
          flush=True)

    # Cleanup: if we loaded the model ourselves, free it. If the caller
    # preloaded it (to reuse across multiple mask probes), leave it alone.
    extractor.remove()
    del extractor, activations, sampled_acts
    if preloaded is None:
        del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return prop_results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve checkpoint list
    # ------------------------------------------------------------------
    if not args.ckpts_json:
        raise SystemExit(
            "--ckpts_json is required. Pass a JSON list of {label, path} "
            "entries — see docstring for the expected format."
        )
    with open(args.ckpts_json) as f:
        ckpt_specs = json.load(f)

    resolved = []
    for spec in ckpt_specs:
        path = spec["path"]
        # HF-hub IDs are always "org/name" — skip local-fs existence check
        is_local = os.sep in path or path.startswith(".") or path.startswith("/")
        if is_local and not os.path.exists(path):
            msg = f"[main] Checkpoint missing: {path} ({spec['label']})"
            if args.skip_missing:
                print(f"  WARNING: {msg} — skipping")
                continue
            raise FileNotFoundError(msg)
        resolved.append(spec)

    print(f"[main] Will probe {'baseline + ' if not args.skip_baseline else ''}"
          f"{len(resolved)} trained checkpoints")
    if not args.skip_baseline:
        print(f"       {'Baseline':20s} ← {args.baseline_path}")
    for s in resolved:
        print(f"       {s['label']:20s} ← {s['path']}")

    # ------------------------------------------------------------------
    # Probe datasets + text concatenation
    # ------------------------------------------------------------------
    probe_datasets = load_probe_datasets(args)

    all_texts = []
    property_slices = {}
    for prop_name, prop_data in probe_datasets.items():
        texts = [t for t, _ in prop_data["examples"]]
        property_slices[prop_name] = slice(
            len(all_texts), len(all_texts) + len(texts)
        )
        all_texts.extend(texts)
    print(f"[main] {len(probe_datasets)} properties, "
          f"{len(all_texts)} texts total")

    # ------------------------------------------------------------------
    # Iterate: group specs by checkpoint path, so we can load each ckpt once
    # and probe it with multiple masks (sub-network ablation). Without this,
    # a 3-mask × 3-ckpt matrix would do 9 from_pretrained() calls.
    # ------------------------------------------------------------------
    results_by_config = {}
    sample_indices_ref = {"resolved": None, "n_mlp": None}

    # Build ordered list of (label, path, mask_path)
    work = []
    if not args.skip_baseline:
        work.append((args.baseline_label, args.baseline_path, None))
    for spec in resolved:
        work.append((spec["label"], spec["path"], spec.get("mask_path")))

    # Group consecutive runs by ckpt path so we can preload. Re-ordering
    # would change visual order; instead, we detect runs where adjacent
    # entries share the ckpt and reuse the model across them.
    i = 0
    while i < len(work):
        ckpt_path = work[i][1]
        # collect consecutive entries on same ckpt
        j = i
        while j < len(work) and work[j][1] == ckpt_path:
            j += 1
        group = work[i:j]

        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        print(f"\n[main] Loaded {ckpt_path} in {time.time() - t0:.1f}s; "
              f"will probe with {len(group)} mask config(s)")

        for (label, _p, mask_path) in group:
            results_by_config[label] = evaluate_checkpoint(
                ckpt_path, label, all_texts, args,
                property_slices, probe_datasets, sample_indices_ref,
                mask_path=mask_path,
                preloaded=(model, tokenizer),
            )
            with open(os.path.join(args.output_dir, "probe_results.json"), "w") as f:
                json.dump(results_by_config, f, indent=2)

        # Done with this ckpt; free before loading the next.
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        i = j

    # ------------------------------------------------------------------
    # Final save + plots
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(results_by_config, f, indent=2)
    print(f"\n[main] Saved results JSON → {json_path}")

    if not results_by_config:
        print("[main] No configs evaluated — skipping plots.")
        return

    sample_indices = sample_indices_ref["resolved"] or []

    plot_probe_heatmap(
        results_by_config,
        sample_indices=sample_indices,
        property_order=list(probe_datasets.keys()),
        output_path=os.path.join(args.output_dir, "probe_heatmap_all.png"),
    )

    if (not args.skip_baseline
            and args.baseline_label in results_by_config
            and len(results_by_config) > 1):
        plot_delta_heatmap(
            results_by_config,
            baseline_key=args.baseline_label,
            sample_indices=sample_indices,
            property_order=list(probe_datasets.keys()),
            output_path=os.path.join(args.output_dir, "probe_delta_all.png"),
        )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the pairwise-ranking probe over baseline + 12 masks in a single job.

Baseline activations are collected exactly once. Each mask re-uses the
already-loaded model + tokenizer + cached probe dataset, so the only extra
cost per mask is one inference pass over the probe texts.

Output: one JSON with 13 configs (baseline + 12) keyed by display label,
plus absolute and Δ heatmaps.

Usage (cluster default paths):
    python src/analysis/probe_pair_12masks.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir /scratch/$USER/probe_pair_12masks/

Override masks via JSON list:
    python src/analysis/probe_pair_12masks.py \\
        --masks_json my_masks.json ...
where my_masks.json = [{"label": "Cold-SNIP-DPO", "path": "/.../cold_snip_dpo.pt"}, ...]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
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


# ---------------------------------------------------------------------------
# Default 12-mask layout (matches plot_12mask_probe.py panel order)
# ---------------------------------------------------------------------------
DEFAULT_MASKS = [
    # Cold-start DPO
    {"label": "Cold-SNIP-DPO",    "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_snip_dpo.pt"},
    {"label": "Cold-CAV-DPO",     "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt"},
    {"label": "Cold-Fisher-DPO",  "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_fisher_dpo.pt"},
    # Warm-start DPO (step50 sp97.5 files)
    {"label": "Warm-Fisher-DPO",    "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_fisher_step50_sp97.5.pt"},
    {"label": "Warm-Magnitude-DPO", "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt"},
    {"label": "Warm-Momentum-DPO",  "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_momentum_step50_sp97.5.pt"},
    # Cold-start GRPO
    {"label": "Cold-SNIP-GRPO",   "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_snip_grpo.pt"},
    {"label": "Cold-CAV-GRPO",    "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_grpo.pt"},
    {"label": "Cold-Fisher-GRPO", "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_fisher_grpo.pt"},
    # Warm-start GRPO
    {"label": "Warm-Fisher-GRPO",    "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_fisher_grpo.pt"},
    {"label": "Warm-Magnitude-GRPO", "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt"},
    {"label": "Warm-Momentum-GRPO",  "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_momentum_grpo.pt"},
]

BASELINE_KEY = "Baseline\n(no mask)"


def parse_args():
    p = argparse.ArgumentParser(
        description="Pairwise ranking probe over baseline + 12 masks"
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--masks_json", default=None,
                   help="Path to JSON list of {label, path}. Defaults to built-in 12.")
    p.add_argument("--output_dir", default="probe_pair_12masks")
    p.add_argument("--layer_stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--pairs_per_pos", type=int, default=2)
    p.add_argument("--samples_per_class", type=int, default=None)
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--probe_cache", default=None)
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip the unmasked baseline (disables delta heatmap)")
    p.add_argument("--skip_missing", action="store_true",
                   help="If a mask file is missing, log a warning and skip instead of failing")
    return p.parse_args()


def load_probe_datasets(args):
    """Load (or build + cache) the HF probe datasets — shared format with probe_analysis.py."""
    cache_dir = args.dataset_cache_dir or os.environ.get("HF_DATASETS_CACHE")
    _CACHE_VERSION = "v2"
    if args.probe_cache:
        probe_cache_path = args.probe_cache
    else:
        n_tag = args.samples_per_class if args.samples_per_class is not None else "all"
        cache_name = f"probe_dataset_cache_{_CACHE_VERSION}_n{n_tag}_seed42.json"
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
            cache_dir=cache_dir,
        )
        os.makedirs(os.path.dirname(probe_cache_path), exist_ok=True)
        with open(probe_cache_path, "w") as f:
            json.dump(datasets, f)
        print(f"[data] Cached probe dataset → {probe_cache_path}")

    # Sanity: equal size, balanced labels
    expected_n = None
    for prop_name, prop_data in datasets.items():
        n = len(prop_data["examples"])
        if expected_n is None:
            expected_n = n
        assert n == expected_n, f"'{prop_name}' has {n}, expected {expected_n}"
        lbls = [l for _, l in prop_data["examples"]]
        assert sum(1 for l in lbls if l == 1) == sum(1 for l in lbls if l == 0)
    return datasets


def evaluate_config(model, tokenizer, extractor, all_texts, args,
                    sample_index_set, property_slices, probe_datasets,
                    mask_dict, label_clean):
    """Run one forward pass + train pairwise probes for a single config."""
    ctx = apply_mask(model, mask_dict) if mask_dict is not None else no_mask()
    t0 = time.time()
    with ctx:
        device = next(model.parameters()).device
        activations = extractor.collect(
            model, tokenizer, all_texts, device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    print(f"  [{label_clean}] activation pass: {time.time() - t0:.1f}s")

    sampled_acts = {
        name: acts for name, acts in activations.items()
        if _layer_index(name) in sample_index_set
    }

    prop_results = {}
    for prop_name, prop_data in probe_datasets.items():
        slc = property_slices[prop_name]
        labels_arr = np.array([lbl for _, lbl in prop_data["examples"]])
        prop_acts = {name: acts[slc] for name, acts in sampled_acts.items()}

        layer_accs = train_pairwise_probes(
            prop_acts, labels_arr,
            cv=args.cv_folds,
            pairs_per_pos=args.pairs_per_pos,
        )
        prop_results[prop_name] = layer_accs

        test_vals = [v["test"] for v in layer_accs.values()]
        train_vals = [v["train"] for v in layer_accs.values()]
        n_notconv = sum(1 for v in layer_accs.values() if not v["converged"])
        n_degen = sum(1 for v in layer_accs.values() if v["degenerate"])
        print(
            f"  {prop_name:12s}: "
            f"test={np.nanmean(test_vals):.3f}  "
            f"train={np.nanmean(train_vals):.3f}  "
            f"gap={np.nanmean(train_vals) - np.nanmean(test_vals):+.3f}  "
            f"non_conv={n_notconv}/{len(layer_accs)}  "
            f"degen={n_degen}/{len(layer_accs)}"
        )
    return prop_results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve mask list
    # ------------------------------------------------------------------
    if args.masks_json:
        with open(args.masks_json) as f:
            mask_specs = json.load(f)
    else:
        mask_specs = DEFAULT_MASKS

    # Filter missing files if requested
    resolved = []
    for spec in mask_specs:
        if not os.path.exists(spec["path"]):
            msg = f"[main] Mask file missing: {spec['path']} ({spec['label']})"
            if args.skip_missing:
                print(f"  WARNING: {msg} — skipping")
                continue
            raise FileNotFoundError(msg)
        resolved.append(spec)

    print(f"[main] Will evaluate {'baseline + ' if not args.skip_baseline else ''}"
          f"{len(resolved)} masks")
    for s in resolved:
        print(f"       {s['label']:24s} ← {s['path']}")

    # ------------------------------------------------------------------
    # Probe datasets + text concatenation
    # ------------------------------------------------------------------
    probe_datasets = load_probe_datasets(args)

    all_texts = []
    property_slices = {}
    for prop_name, prop_data in probe_datasets.items():
        texts = [t for t, _ in prop_data["examples"]]
        property_slices[prop_name] = slice(len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)
    print(f"[main] {len(probe_datasets)} properties, "
          f"{len(all_texts)} texts total")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\n[main] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_mlp_layers = sum(
        1 for name, _ in model.named_modules() if name.endswith("down_proj")
    )
    sample_indices = list(range(0, n_mlp_layers, args.layer_stride))
    if (n_mlp_layers - 1) not in sample_indices:
        sample_indices.append(n_mlp_layers - 1)
    sample_index_set = set(sample_indices)
    print(f"[main] Sampling {len(sample_indices)} layers: {sample_indices}")

    # ------------------------------------------------------------------
    # Load all masks up front (lets us fail fast on bad files)
    # ------------------------------------------------------------------
    loaded_masks = []
    for spec in resolved:
        print(f"[main] Loading {spec['label']}")
        loaded_masks.append((spec["label"], load_mask(spec["path"])))

    # ------------------------------------------------------------------
    # Evaluate baseline once, then each mask
    # ------------------------------------------------------------------
    extractor = FeatureExtractor().register(model)
    results_by_config = {}

    if not args.skip_baseline:
        print(f"\n[main] ===== {BASELINE_KEY.replace(chr(10), ' ')} =====")
        results_by_config[BASELINE_KEY] = evaluate_config(
            model, tokenizer, extractor, all_texts, args,
            sample_index_set, property_slices, probe_datasets,
            mask_dict=None, label_clean="baseline",
        )

    for label, mask_dict in loaded_masks:
        print(f"\n[main] ===== {label} =====")
        results_by_config[label] = evaluate_config(
            model, tokenizer, extractor, all_texts, args,
            sample_index_set, property_slices, probe_datasets,
            mask_dict=mask_dict, label_clean=label,
        )
        # Save after each mask so partial progress survives crashes
        partial_path = os.path.join(args.output_dir, "probe_pair_results.json")
        with open(partial_path, "w") as f:
            json.dump(results_by_config, f, indent=2)

    extractor.remove()

    # ------------------------------------------------------------------
    # Final save + plots
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "probe_pair_results.json")
    with open(json_path, "w") as f:
        json.dump(results_by_config, f, indent=2)
    print(f"\n[main] Saved results JSON → {json_path}")

    if not results_by_config:
        print("[main] No configs evaluated — skipping plots.")
        return

    # Absolute heatmap: too wide with 13 configs, but still produce it
    plot_probe_heatmap(
        results_by_config,
        sample_indices=sample_indices,
        property_order=list(probe_datasets.keys()),
        output_path=os.path.join(args.output_dir, "probe_pair_heatmap_all.png"),
    )

    # Δ plot only makes sense with baseline AND at least one mask
    if BASELINE_KEY in results_by_config and len(results_by_config) > 1:
        plot_delta_heatmap(
            results_by_config,
            baseline_key=BASELINE_KEY,
            sample_indices=sample_indices,
            property_order=list(probe_datasets.keys()),
            output_path=os.path.join(args.output_dir, "probe_pair_delta_all.png"),
        )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()

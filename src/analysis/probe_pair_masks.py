#!/usr/bin/env python3
"""
Run the pairwise-ranking probe over baseline + N masks in a single job.

Baseline activations are collected exactly once. Each mask re-uses the
already-loaded model + tokenizer + cached probe dataset, so the only extra
cost per mask is one inference pass over the probe texts.

Output: one JSON with (baseline + N) configs keyed by display label, plus
absolute and Δ heatmaps.

Default mask set is the 6-config oracle/cav/random study across DPO and
GRPO sides; override with --masks_json to run any other list.

Usage (cluster default paths):
    python src/analysis/probe_pair_masks.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir /scratch/$USER/probe_pair_masks/

Override masks via JSON list:
    python src/analysis/probe_pair_masks.py \\
        --masks_json my_masks.json ...
where my_masks.json = [{"label": "Cold-CAV-DPO", "path": "/.../cold_cav_dpo.pt"}, ...]
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

_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
# cold_start.utils.__init__ eagerly imports SNIPScorer, which uses
# `from src.utils.mask_utils ...` — that needs the repo root on sys.path,
# not just src/. Add both to be safe.
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


# ---------------------------------------------------------------------------
# Default 6-mask layout: Oracle (Warm-Magnitude) / Cold-CAV / Random,
# evaluated on both DPO and GRPO sides. Override with --masks_json.
# ---------------------------------------------------------------------------
DEFAULT_MASKS = [
    # DPO side
    {"label": "Oracle-DPO",    "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt"},
    {"label": "Cold-CAV-DPO",  "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt"},
    {"label": "Random-DPO",    "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b/random_baseline_dpo_sp97.5_seed42.pt"},
    # GRPO side
    {"label": "Oracle-GRPO",   "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt"},
    {"label": "Cold-CAV-GRPO", "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_grpo.pt"},
    {"label": "Random-GRPO",   "path": "/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/random_baseline_grpo_sp97.5_seed42.pt"},
]

BASELINE_KEY = "Baseline\n(no mask)"


def parse_args():
    p = argparse.ArgumentParser(
        description="Pairwise ranking probe over baseline + N masks"
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--masks_json", default=None,
                   help="Path to JSON list of {label, path}. Defaults to "
                        "the built-in 6-mask oracle/cav/random set.")
    p.add_argument("--output_dir", default="probe_pair_masks")
    p.add_argument("--layer_stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--pairs_per_pos", type=int, default=2)
    p.add_argument("--n_jobs", type=int, default=8,
                   help="Parallel workers for per-layer probe training (default: 8)")
    p.add_argument("--holdout_frac", type=float, default=0.0,
                   help="Stratified per-class fraction reserved as untouched "
                        "holdout before CV (default: 0.0 = CV only)")
    p.add_argument("--use_holdout_as_test", action="store_true",
                   help="When --holdout_frac > 0, use holdout accuracy as the "
                        "primary 'test' value used by summaries and plots. "
                        "CV diagnostics remain in cv_* fields.")
    p.add_argument("--probe_C", type=float, default=1.0,
                   help="L2 regularization strength for the logistic probe. "
                        "Lower = stronger (default: 1.0; try 0.1 when adding "
                        "preference probes to curb overfitting).")
    p.add_argument("--samples_per_class", type=int, default=None,
                   help="Per-class cap for the 4 benchmark probes "
                        "(syntax/semantics/factual/math).")
    p.add_argument("--preference_samples_per_class", type=int, default=1000,
                   help="Per-class cap for tulu3/open-r1 preference probes "
                        "(scheme C). Default 1000 gives ~8000 pair rows which "
                        "brings p/n below the overfitting threshold.")
    p.add_argument("--no_preference", action="store_true",
                   help="Disable tulu3/open-r1 preference probes; run only "
                        "the original 4-benchmark setup.")
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--probe_cache", default=None)
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip the unmasked baseline (disables delta heatmap)")
    p.add_argument("--skip_missing", action="store_true",
                   help="If a mask file is missing, log a warning and skip instead of failing")
    p.add_argument("--patch_mode", default="zero_out",
                   choices=["zero_out", "delta_only", "anti_delta_only"],
                   help="How apply_mask edits weights. zero_out (default) = "
                        "w_ft * M (legacy). delta_only = w_base + (w_ft-w_base)*M "
                        "(sufficiency: keep RL Δθ only on M). anti_delta_only = "
                        "w_base + (w_ft-w_base)*(1-M) (necessity: revert M back "
                        "to base, keep Δθ on the complement). The latter two "
                        "require --base_model.")
    p.add_argument("--base_model", default=None,
                   help="HF id or local path of the base (pre-finetune) model. "
                        "Required when --patch_mode is delta_only or "
                        "anti_delta_only. Loaded once on CPU as a state-dict.")
    return p.parse_args()


def load_probe_datasets(args):
    """Load (or build + cache) the HF probe datasets — shared format with probe_analysis.py."""
    cache_dir = args.dataset_cache_dir or os.environ.get("HF_DATASETS_CACHE")
    # v3: adds tulu3 dpo_preference + open-r1 grpo_preference (scheme C).
    # Preference probes have different per-class counts than benchmarks, so
    # old v2 caches (which required equal N across properties) aren't reusable.
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

    # Sanity: each property is pos/neg balanced. Sizes may differ *across*
    # properties now (preference probes use a higher cap than benchmarks), so
    # we no longer require a single expected_n — each property stands alone.
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


def evaluate_config(model, tokenizer, extractor, all_texts, args,
                    sample_index_set, property_slices, probe_datasets,
                    mask_dict, label_clean, base_state_dict=None):
    """Run one forward pass + train pairwise probes for a single config."""
    if mask_dict is None:
        ctx = no_mask()
    else:
        ctx = apply_mask(
            model, mask_dict,
            patch_mode=args.patch_mode,
            base_state_dict=base_state_dict,
        )
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
    cfg_t0 = time.time()
    cfg_notconv = 0
    cfg_degen = 0
    cfg_nonfinite = 0
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

        reason_msg = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        if not reason_msg:
            reason_msg = "none"
        metric_msg = (
            f"  {prop_name:12s} ({prop_dt:5.1f}s): "
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
        print(
            metric_msg,
            flush=True,
        )
    cfg_reason_msg = ", ".join(
        f"{k}:{v}" for k, v in sorted(cfg_reason_counts.items())
    )
    if not cfg_reason_msg:
        cfg_reason_msg = "none"
    print(
        f"  [{label_clean}] diagnostics: "
        f"non_conv={cfg_notconv} degen={cfg_degen} "
        f"nonfinite={cfg_nonfinite} reasons={cfg_reason_msg}",
        flush=True,
    )
    print(f"  [{label_clean}] total probe time: {time.time() - cfg_t0:.1f}s",
          flush=True)
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
    # Optional: load base model state dict for delta_only / anti_delta_only
    # ------------------------------------------------------------------
    base_state_dict = None
    if args.patch_mode in {"delta_only", "anti_delta_only"}:
        if not args.base_model:
            raise ValueError(
                f"--patch_mode={args.patch_mode} requires --base_model "
                f"(pre-finetune checkpoint, e.g. meta-llama/Llama-3.1-8B-Instruct)."
            )
        print(f"\n[main] Loading base model state dict for patch_mode="
              f"{args.patch_mode}: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        # Snapshot only the params we'll touch (intersection with masks). Keeps
        # CPU RAM lean.
        wanted = set()
        for _, md in loaded_masks:
            wanted.update(md.keys())
        base_state_dict = {
            n: p.detach().cpu().clone()
            for n, p in base_model.named_parameters()
            if n in wanted
        }
        print(f"[main] Base state dict snapped: "
              f"{len(base_state_dict)}/{len(wanted)} mask params matched")
        del base_model

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
            base_state_dict=base_state_dict,
        )

    for label, mask_dict in loaded_masks:
        print(f"\n[main] ===== {label} =====")
        results_by_config[label] = evaluate_config(
            model, tokenizer, extractor, all_texts, args,
            sample_index_set, property_slices, probe_datasets,
            mask_dict=mask_dict, label_clean=label,
            base_state_dict=base_state_dict,
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

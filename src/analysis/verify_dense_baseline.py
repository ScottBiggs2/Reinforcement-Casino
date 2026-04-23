#!/usr/bin/env python3
"""Sanity-check the dense-baseline pairwise probe.

Produces three numbers per (property, layer) for the *unmasked* model:

  - real_test   : pairwise accuracy on a completely untouched holdout split
  - real_train  : training-pair accuracy (gap vs real_test = overfitting signal)
  - shuffle_test: same probe trained on RANDOMLY SHUFFLED labels
                  (must collapse to ~0.5 if the probe isn't cheating)

If real_test is high (~0.9) AND shuffle_test is ~0.5 AND real_train-real_test
is small, the dense-baseline heatmap is measuring real structure, not a
confound / leak / overfitting.

Outputs go to --output_dir (default: /scratch/$USER/probe_verify_dense).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from cold_start.utils.activation_hooks import FeatureExtractor
from analysis.probe_analysis import _layer_index, _load_hf_probe_datasets, no_mask
from analysis.probe_analysis_pair import train_pairwise_probes


def _default_output_dir() -> str:
    user = os.environ.get("USER", "anon")
    return f"/scratch/{user}/probe_verify_dense"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", default=_default_output_dir())
    p.add_argument("--layer_stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--pairs_per_pos", type=int, default=2)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--samples_per_class", type=int, default=None)
    p.add_argument("--shuffle_seed", type=int, default=1337)
    p.add_argument("--probe_cache", default=None,
                   help="Reuse an existing probe_dataset_cache_*.json if passed.")
    return p.parse_args()


def _run_probe(prop_acts, labels, args, shuffle: bool):
    if shuffle:
        rng = np.random.default_rng(args.shuffle_seed)
        labels = rng.permutation(labels)
    return train_pairwise_probes(
        prop_acts, labels,
        cv=args.cv_folds,
        pairs_per_pos=args.pairs_per_pos,
        n_jobs=args.n_jobs,
        holdout_frac=args.holdout_frac,
        use_holdout_as_test=True,
        seed=42 if not shuffle else args.shuffle_seed,
    )


def _plot(results: dict, out_path: str):
    props = list(results.keys())
    fig, axes = plt.subplots(1, len(props), figsize=(4.2 * len(props), 3.6),
                             sharey=True)
    if len(props) == 1:
        axes = [axes]
    for ax, prop in zip(axes, props):
        layers = sorted(results[prop]["real"].keys(), key=lambda n: _layer_index(n))
        xs = [_layer_index(n) for n in layers]
        real_test = [results[prop]["real"][n]["test"] for n in layers]
        real_train = [results[prop]["real"][n]["train"] for n in layers]
        shuf_test = [results[prop]["shuffle"][n]["test"] for n in layers]
        ax.plot(xs, real_train, "o--", color="#1f77b4", label="real-train", alpha=0.6)
        ax.plot(xs, real_test, "o-", color="#1f77b4", label="real-test (holdout)")
        ax.plot(xs, shuf_test, "s-", color="#d62728", label="shuffled-labels test")
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("MLP layer")
        ax.set_title(prop)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pairwise probe accuracy")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Dense-baseline verification (real vs shuffled labels, holdout)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[verify] plot → {out_path}")


def _summary_verdict(results: dict) -> list:
    lines = []
    for prop, blocks in results.items():
        real = np.array([v["test"] for v in blocks["real"].values()])
        train = np.array([v["train"] for v in blocks["real"].values()])
        shuf = np.array([v["test"] for v in blocks["shuffle"].values()])
        real_m = float(np.nanmean(real))
        gap_m = float(np.nanmean(train - real))
        shuf_m = float(np.nanmean(shuf))
        shuf_max = float(np.nanmax(shuf))
        ok_signal = real_m > 0.75
        ok_nocheat = shuf_max < 0.60
        ok_nofit = gap_m < 0.10
        verdict = "PASS" if (ok_signal and ok_nocheat and ok_nofit) else "CHECK"
        lines.append(
            f"  {prop:20s} real={real_m:.3f}  train-test gap={gap_m:+.3f}  "
            f"shuffle mean/max={shuf_m:.3f}/{shuf_max:.3f}  → {verdict}"
        )
    return lines


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- probe dataset (reuse existing cache if given) -------------------
    if args.probe_cache and os.path.exists(args.probe_cache):
        print(f"[data] reusing cache: {args.probe_cache}")
        with open(args.probe_cache) as f:
            probe = json.load(f)
        for name in probe:
            probe[name]["examples"] = [(t, l) for t, l in probe[name]["examples"]]
    else:
        probe = _load_hf_probe_datasets(samples_per_class=args.samples_per_class)

    # --- flatten texts (property_slice trick matches probe_analysis_pair) ---
    all_texts, property_slices = [], {}
    for name, data in probe.items():
        texts = [t for t, _ in data["examples"]]
        property_slices[name] = slice(len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)

    print(f"[verify] {len(probe)} properties, {len(all_texts)} texts total")

    # --- model + activations -------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    n_mlp = sum(1 for n, _ in model.named_modules() if n.endswith("down_proj"))
    sample_idx = list(range(0, n_mlp, args.layer_stride))
    if (n_mlp - 1) not in sample_idx:
        sample_idx.append(n_mlp - 1)
    sample_set = set(sample_idx)
    print(f"[verify] sampling layers {sample_idx}")

    extractor = FeatureExtractor().register(model)
    with no_mask():
        device = next(model.parameters()).device
        activations = extractor.collect(
            model, tok, all_texts, device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
    sampled = {n: a for n, a in activations.items() if _layer_index(n) in sample_set}

    # --- run the two probes per property --------------------------------
    results = {}
    for prop, data in probe.items():
        slc = property_slices[prop]
        labels = np.array([l for _, l in data["examples"]])
        prop_acts = {n: a[slc] for n, a in sampled.items()}

        print(f"\n[verify] ===== {prop} =====")
        real = _run_probe(prop_acts, labels, args, shuffle=False)
        shuf = _run_probe(prop_acts, labels, args, shuffle=True)
        results[prop] = {"real": real, "shuffle": shuf}

    # --- save ----------------------------------------------------------
    json_path = os.path.join(args.output_dir, "verify_dense_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[verify] json → {json_path}")

    png_path = os.path.join(args.output_dir, "verify_dense.png")
    _plot(results, png_path)

    print("\n=== dense-baseline verdict ===")
    print("PASS = real>0.75 AND shuffle_max<0.60 AND gap<0.10")
    for line in _summary_verdict(results):
        print(line)


if __name__ == "__main__":
    main()

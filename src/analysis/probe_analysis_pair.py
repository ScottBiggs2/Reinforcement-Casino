#!/usr/bin/env python3
"""
Pairwise-ranking probing classifier analysis for sparse mask comparison.

Same pipeline as probe_analysis.py, but the per-layer probe is replaced with
a pairwise Bradley-Terry / reward-model style probe:

    Train w such that  w·h+ > w·h-
    Loss:              -log σ(w·h+ - w·h-)

Equivalent to logistic regression on activation differences (h+ - h-) with
no intercept. Accuracy is the fraction of held-out (pos, neg) pairs for
which w·h+ > w·h-. This mirrors how DPO / reward models learn a "good
direction" — the probe tells us whether each layer's activations contain
a linear direction that separates positive from negative prompts.

Usage:
    python src/analysis/probe_analysis_pair.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --mask_a masks/cold_snip_llama_90pct.pt \\
        --mask_b masks/warm_momentum_llama_90pct.pt \\
        --mask_a_label "SNIP" \\
        --mask_b_label "Momentum" \\
        --output_dir probe_results/
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).parent.parent
_ROOT = _SRC.parent
# Both: src/ lets `from cold_start...` work; repo root lets the snip_scorer
# chain (`from src.utils.mask_utils`) resolve when cold_start.utils.__init__
# pulls in SNIPScorer eagerly.
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


# ---------------------------------------------------------------------------
# Pairwise ranking probe
# ---------------------------------------------------------------------------

def _build_pairs(pos_idx: np.ndarray, neg_idx: np.ndarray,
                 pairs_per_pos: int, rng: np.random.Generator) -> np.ndarray:
    """Build (pos, neg) index pairs by randomly matching each pos to K negs.

    Returns array of shape [len(pos_idx) * pairs_per_pos, 2] with columns
    (pos_row, neg_row).
    """
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = []
    for p in pos_idx:
        # Sample with replacement when neg pool is smaller than pairs_per_pos
        replace = len(neg_idx) < pairs_per_pos
        sampled = rng.choice(neg_idx, size=pairs_per_pos, replace=replace)
        for n in sampled:
            pairs.append((p, n))
    return np.array(pairs, dtype=np.int64)


def _train_one_layer(
    layer_name: str,
    X: np.ndarray,
    pos_idx_all: np.ndarray,
    neg_idx_all: np.ndarray,
    pos_splits: list,
    neg_splits: list,
    pairs_per_pos: int,
    seed: int,
) -> tuple:
    """Fit all folds for one layer. Returned as (name, result_dict) so
    Parallel(...) output can be converted straight to a dict."""
    # Only bail out on genuinely unrecoverable inputs: non-finite values
    # (±inf or NaN would make LR explode) or an exactly-constant feature
    # matrix (StandardScaler divides by zero). Very small std (e.g. 1e-5
    # on a heavily-pruned network) is fine — the probe will simply
    # produce accuracy ≈0.5, which is itself the answer ("no linearly
    # separable direction exists here"). The old 1e-8 threshold was too
    # conservative and masked every high-sparsity warm mask as NaN.
    if not np.isfinite(X).all() or float(X.std()) < 1e-16:
        return layer_name, {
            "test": float("nan"), "std": float("nan"),
            "train": float("nan"), "gap": float("nan"),
            "converged": False, "degenerate": True,
        }

    rng = np.random.default_rng(seed + hash(layer_name) % (2**31))
    test_accs, train_accs = [], []
    all_converged = True

    for (p_tr, p_te), (n_tr, n_te) in zip(pos_splits, neg_splits):
        pos_tr = pos_idx_all[p_tr]
        pos_te = pos_idx_all[p_te]
        neg_tr = neg_idx_all[n_tr]
        neg_te = neg_idx_all[n_te]

        train_pairs = _build_pairs(pos_tr, neg_tr, pairs_per_pos, rng)
        test_pairs = _build_pairs(pos_te, neg_te, pairs_per_pos, rng)

        # Symmetric dataset on diffs: (h+ - h-, 1) and (h- - h+, 0).
        diff_pos = X[train_pairs[:, 0]] - X[train_pairs[:, 1]]
        X_train = np.concatenate([diff_pos, -diff_pos], axis=0)
        y_train = np.concatenate([np.ones(len(diff_pos)), np.zeros(len(diff_pos))])

        diff_te = X[test_pairs[:, 0]] - X[test_pairs[:, 1]]
        X_test = np.concatenate([diff_te, -diff_te], axis=0)
        y_test = np.concatenate([np.ones(len(diff_te)), np.zeros(len(diff_te))])

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(
                penalty="l2", solver="lbfgs", C=1.0,
                max_iter=2000, fit_intercept=False, random_state=seed,
            )),
        ])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(X_train, y_train)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                all_converged = False

        train_accs.append(pipe.score(X_train, y_train))
        test_accs.append(pipe.score(X_test, y_test))

    test_arr = np.array(test_accs)
    train_arr = np.array(train_accs)
    return layer_name, {
        "test": float(test_arr.mean()),
        "std": float(test_arr.std()),
        "train": float(train_arr.mean()),
        "gap": float(train_arr.mean() - test_arr.mean()),
        "converged": all_converged,
        "degenerate": False,
    }


def train_pairwise_probes(
    activations_by_layer: dict,
    labels: np.ndarray,
    cv: int = 5,
    pairs_per_pos: int = 2,
    seed: int = 42,
    n_jobs: int = 8,
) -> dict:
    """Train a pairwise ranking probe per layer (parallel over layers).

    Splits pos and neg samples into K folds *separately* so no sample
    appears in both train and test pairs. Pairs are (pos, neg); for each
    pair we create the signed difference h+ - h- (label 1) and its negation
    (label 0), then fit an L2 logistic regression with no intercept. The
    learned coefficient vector is the ranking direction w; cross-validated
    accuracy is the fraction of test pairs ranked correctly.

    Layers are trained in parallel (joblib.Parallel), which is the main
    speedup lever: per-layer LR on 14336-dim features dominates runtime.
    With n_jobs=8 this is ~6-8× faster than the sequential version.
    """
    if pairs_per_pos < 1:
        raise ValueError(f"pairs_per_pos must be >= 1, got {pairs_per_pos}")

    pos_idx_all = np.where(labels == 1)[0]
    neg_idx_all = np.where(labels == 0)[0]

    # Guard: KFold crashes if n_splits > n_samples. Fall back to min class size.
    effective_cv = min(cv, len(pos_idx_all), len(neg_idx_all))
    if effective_cv < 2:
        raise ValueError(
            f"Need >=2 samples per class for CV, got "
            f"pos={len(pos_idx_all)}, neg={len(neg_idx_all)}"
        )
    if effective_cv < cv:
        print(f"[pair] WARNING: reducing cv from {cv} to {effective_cv} "
              f"(pos={len(pos_idx_all)}, neg={len(neg_idx_all)})")

    kf_pos = KFold(n_splits=effective_cv, shuffle=True, random_state=seed)
    kf_neg = KFold(n_splits=effective_cv, shuffle=True, random_state=seed + 1)
    pos_splits = list(kf_pos.split(pos_idx_all))
    neg_splits = list(kf_neg.split(neg_idx_all))

    # Convert activations to numpy once, outside the parallel section, so
    # worker processes receive already-materialized arrays (joblib with
    # loky backend memmaps large arrays automatically).
    layer_items = [
        (name, acts.float().numpy()) for name, acts in activations_by_layer.items()
    ]

    # Fast path: skip Parallel's fork overhead entirely when serial is asked.
    if n_jobs == 1 or len(layer_items) <= 1:
        results = [
            _train_one_layer(
                name, X, pos_idx_all, neg_idx_all,
                pos_splits, neg_splits, pairs_per_pos, seed,
            )
            for name, X in layer_items
        ]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_train_one_layer)(
                name, X, pos_idx_all, neg_idx_all,
                pos_splits, neg_splits, pairs_per_pos, seed,
            )
            for name, X in layer_items
        )
    return dict(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pairwise ranking probe analysis for mask comparison"
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--mask_a", required=True)
    p.add_argument("--mask_b", required=True)
    p.add_argument("--mask_a_label", default="Mask A")
    p.add_argument("--mask_b_label", default="Mask B")
    p.add_argument("--include_baseline", action="store_true", default=True)
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--output_dir", default="probe_pair_results")
    p.add_argument("--layer_stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--pairs_per_pos", type=int, default=2,
                   help="Number of (pos, neg) pairs to sample per positive "
                        "example per fold (default: 2)")
    p.add_argument("--n_jobs", type=int, default=8,
                   help="Parallel workers for per-layer probe training (default: 8)")
    p.add_argument("--samples_per_class", type=int, default=None)
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--probe_cache", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_baseline:
        args.include_baseline = False
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load probe datasets (shared cache format with probe_analysis.py)
    # ------------------------------------------------------------------
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
            PROBE_DATASETS = json.load(f)
        for prop in PROBE_DATASETS:
            PROBE_DATASETS[prop]["examples"] = [
                (t, l) for t, l in PROBE_DATASETS[prop]["examples"]
            ]
        print(f"[data] Loaded {len(PROBE_DATASETS)} properties from cache")
    else:
        PROBE_DATASETS = _load_hf_probe_datasets(
            samples_per_class=args.samples_per_class,
            cache_dir=cache_dir,
        )
        os.makedirs(os.path.dirname(probe_cache_path), exist_ok=True)
        with open(probe_cache_path, "w") as f:
            json.dump(PROBE_DATASETS, f)
        print(f"[data] Cached probe dataset → {probe_cache_path}")

    expected_n = None
    for prop_name, prop_data in PROBE_DATASETS.items():
        n = len(prop_data["examples"])
        if expected_n is None:
            expected_n = n
        assert n == expected_n
        labels = [lbl for _, lbl in prop_data["examples"]]
        assert sum(1 for l in labels if l == 1) == sum(1 for l in labels if l == 0)

    # ------------------------------------------------------------------
    # Concatenate texts
    # ------------------------------------------------------------------
    all_texts = []
    property_slices = {}
    for prop_name, prop_data in PROBE_DATASETS.items():
        texts = [t for t, _ in prop_data["examples"]]
        property_slices[prop_name] = slice(len(all_texts), len(all_texts) + len(texts))
        all_texts.extend(texts)

    print(f"[main] {len(PROBE_DATASETS)} probe properties, "
          f"{expected_n} examples each, {len(all_texts)} texts total")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n[main] Loading tokenizer & model: {args.model}")
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
    print(f"[main] Model has {n_mlp_layers} MLP layers; "
          f"sampling {len(sample_indices)} layers: {sample_indices}")

    # ------------------------------------------------------------------
    # Load masks
    # ------------------------------------------------------------------
    print(f"\n[main] Loading mask A: {args.mask_a}")
    mask_a = load_mask(args.mask_a)
    print(f"[main] Loading mask B: {args.mask_b}")
    mask_b = load_mask(args.mask_b)

    configs = {}
    if args.include_baseline:
        configs["Baseline\n(no mask)"] = None
    configs[args.mask_a_label] = mask_a
    configs[args.mask_b_label] = mask_b

    # ------------------------------------------------------------------
    # Activation collection + pairwise probe training
    # ------------------------------------------------------------------
    extractor = FeatureExtractor().register(model)
    results_by_config = {}

    for config_label, mask_dict in configs.items():
        label_clean = config_label.replace("\n", " ")
        print(f"\n[main] ===== {label_clean} =====")

        ctx = apply_mask(model, mask_dict) if mask_dict is not None else no_mask()

        with ctx:
            device = next(model.parameters()).device
            activations = extractor.collect(
                model, tokenizer, all_texts, device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

        sampled_acts = {
            name: acts
            for name, acts in activations.items()
            if _layer_index(name) in sample_index_set
        }

        prop_results = {}
        for prop_name, prop_data in PROBE_DATASETS.items():
            slc = property_slices[prop_name]
            labels_arr = np.array([lbl for _, lbl in prop_data["examples"]])
            prop_acts = {name: acts[slc] for name, acts in sampled_acts.items()}

            layer_accs = train_pairwise_probes(
                prop_acts, labels_arr,
                cv=args.cv_folds,
                pairs_per_pos=args.pairs_per_pos,
                n_jobs=args.n_jobs,
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
                f"[min={np.nanmin(test_vals):.3f} max={np.nanmax(test_vals):.3f}]  "
                f"non_conv={n_notconv}/{len(layer_accs)}  "
                f"degen={n_degen}/{len(layer_accs)}"
            )

        results_by_config[config_label] = prop_results

        total_probes = sum(len(v) for v in prop_results.values())
        total_notconv = sum(
            1 for prop in prop_results.values()
            for v in prop.values() if not v["converged"]
        )
        total_degen = sum(
            1 for prop in prop_results.values()
            for v in prop.values() if v["degenerate"]
        )
        if total_notconv or total_degen:
            print(
                f"  ⚠️  [{label_clean}] {total_notconv}/{total_probes} did not converge, "
                f"{total_degen}/{total_probes} degenerate"
            )

    extractor.remove()

    # ------------------------------------------------------------------
    # Save + plot
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "probe_pair_results.json")
    with open(json_path, "w") as f:
        json.dump(results_by_config, f, indent=2)
    print(f"\n[main] Saved results JSON → {json_path}")

    plot_path = os.path.join(args.output_dir, "probe_pair_heatmap.png")
    plot_probe_heatmap(
        results_by_config,
        sample_indices=sample_indices,
        property_order=list(PROBE_DATASETS.keys()),
        output_path=plot_path,
    )

    baseline_key = "Baseline\n(no mask)"
    if baseline_key in results_by_config:
        delta_path = os.path.join(args.output_dir, "probe_pair_delta_heatmap.png")
        plot_delta_heatmap(
            results_by_config,
            baseline_key=baseline_key,
            sample_indices=sample_indices,
            property_order=list(PROBE_DATASETS.keys()),
            output_path=delta_path,
        )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()

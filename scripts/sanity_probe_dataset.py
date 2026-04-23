#!/usr/bin/env python3
"""Sanity checks for probe datasets.

Checks included:
1) Random-label control (should be near chance)
2) Lexical baseline from shallow text features
3) Multi-seed stability summary

Usage example:
    python scripts/sanity_probe_dataset.py \
        --probe_cache /scratch/$USER/probe_pair_masks/probe_dataset_cache_v2_nall_seed42.json \
        --output_dir /scratch/$USER/probe_pair_masks_sanity
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from analysis.probe_analysis import _load_hf_probe_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Probe dataset sanity checks")
    p.add_argument("--probe_cache", default=None,
                   help="Existing probe_dataset_cache*.json. If missing, dataset will be built.")
    p.add_argument("--output_dir", default="probe_dataset_sanity")
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--samples_per_class", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--max_features", type=int, default=20000)
    return p.parse_args()


def load_probe_datasets(args):
    cache_dir = args.dataset_cache_dir or os.environ.get("HF_DATASETS_CACHE")

    if args.probe_cache and os.path.exists(args.probe_cache):
        print(f"[data] loading cache: {args.probe_cache}")
        with open(args.probe_cache) as f:
            datasets = json.load(f)
        for prop in datasets:
            datasets[prop]["examples"] = [(t, l) for t, l in datasets[prop]["examples"]]
        return datasets

    datasets = _load_hf_probe_datasets(
        samples_per_class=args.samples_per_class,
        cache_dir=cache_dir,
    )

    if args.probe_cache:
        out = args.probe_cache
    else:
        n_tag = args.samples_per_class if args.samples_per_class is not None else "all"
        out = os.path.join(args.output_dir, f"probe_dataset_cache_v2_n{n_tag}_seed42.json")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(datasets, f)
    print(f"[data] cached dataset to: {out}")
    return datasets


def make_shallow_features(texts):
    feats = []
    punct = set(".,!?;:\"'()[]{}-_/\\")
    for t in texts:
        s = t or ""
        n_chars = max(len(s), 1)
        n_words = len(s.split())
        n_digits = sum(ch.isdigit() for ch in s)
        n_punct = sum(ch in punct for ch in s)
        n_upper = sum(ch.isupper() for ch in s)
        feats.append([
            float(n_chars),
            float(n_words),
            float(n_digits) / float(n_chars),
            float(n_punct) / float(n_chars),
            float(n_upper) / float(n_chars),
        ])
    return np.asarray(feats, dtype=np.float64)


def eval_one_seed(texts, labels, seed, test_size, max_features):
    y = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr_idx, te_idx), = splitter.split(np.zeros(len(y)), y)

    tr_texts = [texts[i] for i in tr_idx]
    te_texts = [texts[i] for i in te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    # Shallow lexical baseline (length/punctuation/case style features)
    X_tr_shallow = make_shallow_features(tr_texts)
    X_te_shallow = make_shallow_features(te_texts)
    shallow = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    shallow.fit(X_tr_shallow, y_tr)
    shallow_acc = float(accuracy_score(y_te, shallow.predict(X_te_shallow)))

    # Strong lexical baseline (tf-idf + logistic)
    lexical = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
        )),
        ("lr", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    lexical.fit(tr_texts, y_tr)
    lexical_acc = float(accuracy_score(y_te, lexical.predict(te_texts)))

    # Random-label control on the same lexical model
    rng = np.random.default_rng(seed + 1000003)
    y_perm = rng.permutation(y_tr)
    random_label = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
        )),
        ("lr", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    random_label.fit(tr_texts, y_perm)
    random_label_acc = float(accuracy_score(y_te, random_label.predict(te_texts)))

    majority_acc = float(max(np.mean(y_te == 0), np.mean(y_te == 1)))

    return {
        "seed": int(seed),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "majority_acc": majority_acc,
        "shallow_lexical_acc": shallow_acc,
        "tfidf_lexical_acc": lexical_acc,
        "random_label_acc": random_label_acc,
    }


def agg(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_property(name, examples, args):
    texts = [t for t, _ in examples]
    labels = [int(l) for _, l in examples]

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    by_seed = [
        eval_one_seed(texts, labels, s, args.test_size, args.max_features)
        for s in seeds
    ]

    majority = [x["majority_acc"] for x in by_seed]
    shallow = [x["shallow_lexical_acc"] for x in by_seed]
    tfidf = [x["tfidf_lexical_acc"] for x in by_seed]
    rand = [x["random_label_acc"] for x in by_seed]

    return {
        "n_examples": int(len(examples)),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
        "by_seed": by_seed,
        "aggregate": {
            "majority_acc": agg(majority),
            "shallow_lexical_acc": agg(shallow),
            "tfidf_lexical_acc": agg(tfidf),
            "random_label_acc": agg(rand),
            "lexical_minus_random_mean": float(np.mean(np.asarray(tfidf) - np.asarray(rand))),
            "lexical_minus_majority_mean": float(np.mean(np.asarray(tfidf) - np.asarray(majority))),
        },
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed_start)
    np.random.seed(args.seed_start)

    datasets = load_probe_datasets(args)

    report = {
        "config": {
            "probe_cache": args.probe_cache,
            "output_dir": args.output_dir,
            "n_seeds": args.n_seeds,
            "seed_start": args.seed_start,
            "test_size": args.test_size,
            "max_features": args.max_features,
            "samples_per_class": args.samples_per_class,
        },
        "properties": {},
    }

    print("\n=== Probe Dataset Sanity Summary ===")
    for prop, payload in datasets.items():
        print(f"[run] {prop}")
        summary = summarize_property(prop, payload["examples"], args)
        report["properties"][prop] = summary

        a = summary["aggregate"]
        print(
            f"  n={summary['n_examples']} "
            f"maj={a['majority_acc']['mean']:.3f} "
            f"shallow={a['shallow_lexical_acc']['mean']:.3f} "
            f"tfidf={a['tfidf_lexical_acc']['mean']:.3f} "
            f"rand_lbl={a['random_label_acc']['mean']:.3f} "
            f"tfidf-rand={a['lexical_minus_random_mean']:+.3f}"
        )

    out_json = os.path.join(args.output_dir, "probe_dataset_sanity.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[done] wrote: {out_json}")


if __name__ == "__main__":
    main()

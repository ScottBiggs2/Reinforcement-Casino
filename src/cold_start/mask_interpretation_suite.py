#!/usr/bin/env python3
"""Run pairwise mask interpretation for N checkpoints: Jaccard, optional CKA, CSV export, rollup."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cold_start.mask_jaccard_aggregates import extended_jaccard_report
from src.cold_start.mask_to_jaccard import compute_jaccard, load_masks


def _safe_label(path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return Path(path).stem


def _run_cka_pair(
    py: str,
    mask_a: str,
    mask_b: str,
    out_json: Path,
    *,
    model_name: str,
    dataset_name: str,
    device: str,
    n_samples: int,
    batch_size: int,
) -> bool:
    cka_py = _REPO_ROOT / "src" / "cold_start" / "mask_to_cka.py"
    cmd = [
        py,
        str(cka_py),
        mask_a,
        mask_b,
        "--model_name",
        model_name,
        "--dataset_name",
        dataset_name,
        "--device",
        device,
        "--n_samples",
        str(n_samples),
        "--batch_size",
        str(batch_size),
        "-o",
        str(out_json),
    ]
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return r.returncode == 0


def _run_export_csv(
    py: str,
    mask_a: str,
    mask_b: str,
    out_csv: Path,
    jaccard_json: Optional[Path],
    cka_json: Optional[Path],
    skip_effective_rank: bool,
) -> bool:
    exp_py = _REPO_ROOT / "src" / "cold_start" / "export_layer_metrics_csv.py"
    cmd = [py, str(exp_py), mask_a, mask_b, "-o", str(out_csv)]
    if jaccard_json and jaccard_json.is_file():
        cmd.extend(["--jaccard-json", str(jaccard_json)])
    if cka_json and cka_json.is_file():
        cmd.extend(["--cka-json", str(cka_json)])
    if skip_effective_rank:
        cmd.append("--skip_effective_rank")
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return r.returncode == 0


def _write_jaccard_json(
    mask_a: str,
    mask_b: str,
    out_path: Path,
    device: str,
    extended: str,
) -> Dict[str, Any]:
    masks_a, meta_a = load_masks(mask_a)
    masks_b, meta_b = load_masks(mask_b)
    jaccard = compute_jaccard(masks_a, masks_b, device=device)
    report: Dict[str, Any] = {
        "mask_a": os.path.abspath(mask_a),
        "mask_b": os.path.abspath(mask_b),
        "jaccard": {
            "aggregate": jaccard["aggregate_jaccard"],
            "mean": jaccard["mean_jaccard"],
            "min": jaccard["min_jaccard"],
            "max": jaccard["max_jaccard"],
            "n_layers": jaccard.get("n_layers"),
            "total_intersection": jaccard.get("total_intersection"),
            "total_union": jaccard.get("total_union"),
        },
        "per_layer_jaccard": jaccard["per_layer"],
    }
    if meta_a:
        report["metadata_a"] = meta_a
    if meta_b:
        report["metadata_b"] = meta_b
    if extended != "none":
        ex = extended_jaccard_report(
            masks_a,
            masks_b,
            device=device,
            include_param_buckets=extended in ("param_bucket", "both"),
            include_decoder_layers=extended in ("decoder_layer", "both"),
        )
        if "by_param_bucket" in ex:
            report["jaccard_by_param_bucket"] = ex["by_param_bucket"]
        if "by_decoder_layer" in ex:
            report["jaccard_by_decoder_layer"] = ex["by_decoder_layer"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def _maybe_heatmap(
    labels: Sequence[str],
    matrix: List[List[float]],
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skip heatmap", file=sys.stderr)
        return
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.5)))
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Aggregate Jaccard (pairwise)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote heatmap {out_png}", flush=True)


def _run_plot_layer_metrics(pair_dir: Path, plot_dir: Path, py: str) -> None:
    """2x2 diagnostic PNGs from layer_metrics CSVs (optional; smoke / full bells)."""
    plot_py = _REPO_ROOT / "src/cold_start/plot_layer_metrics_csv.py"
    plot_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            py,
            str(plot_py),
            "--input-dir",
            str(pair_dir),
            "--pattern",
            "layer_metrics_*.csv",
            "--output-dir",
            str(plot_dir),
            "--jaccard-mc-trials",
            "200",
            "--jaccard-mc-seed",
            "42",
            "--y-scale",
            "linear",
        ],
        cwd=str(_REPO_ROOT),
    )
    if r.returncode != 0:
        print(
            "[suite] plot_layer_metrics_csv failed (non-fatal); CSVs are still valid.",
            file=sys.stderr,
        )
    else:
        print(f"[suite] Wrote plot_layer_metrics PNGs under {plot_dir}", flush=True)


def _materialize_smoke_masks(
    reference_pt: str,
    out_dir: Path,
    sparsity: float,
    seed_a: int,
    seed_b: int,
) -> Tuple[List[str], List[str]]:
    """Two iid random masks matching ``reference_pt`` topology; distinct seeds required."""
    from src.utils.mask_utils import save_masks
    from src.warm_start.random_mask_baseline import (
        expected_random_jaccard,
        generate_random_mask,
    )

    if seed_a == seed_b:
        raise ValueError("smoke mode requires two different random seeds")

    ref_masks, ref_meta = load_masks(reference_pt)
    smoke_dir = out_dir / "smoke_masks"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    exp_j = expected_random_jaccard(sparsity)
    base_meta: Dict[str, Any] = {
        "smoke_debug": True,
        "smoke_reference": os.path.abspath(reference_pt),
        "sparsity_percent": sparsity,
        "expected_jaccard_iid_random_pair": round(exp_j, 8),
    }
    if ref_meta:
        keys = list(ref_meta.keys())[:12]
        base_meta["reference_metadata_keys_sample"] = keys

    paths: List[str] = []
    labels: List[str] = []
    for tag, seed in (("a", seed_a), ("b", seed_b)):
        masks = generate_random_mask(ref_masks, sparsity, seed=seed)
        path = smoke_dir / f"smoke_random_{tag}_seed{seed}.pt"
        md = {**base_meta, "smoke_variant": tag, "random_seed": seed}
        save_masks(masks, str(path), md)
        ap = os.path.abspath(str(path))
        paths.append(ap)
        labels.append(f"smoke_rand_{tag}_s{seed}")
    readme = smoke_dir / "README_SMOKE.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "Smoke masks: same topology as smoke_reference, uniform random scores, "
            f"target sparsity {sparsity}%. Expected aggregate Jaccard for two iid random "
            f"masks at this sparsity (closed form): ~{exp_j:.6f}.\n"
        )
    print(f"[smoke] Wrote {paths[0]} and {paths[1]}", flush=True)
    return paths, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpretation suite: all pairs from N mask .pt files (Jaccard JSON, optional CKA, layer_metrics CSV, rollup).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "masks",
        nargs="*",
        default=[],
        help="Mask checkpoint paths (.pt). Omit when using --smoke-debug (two masks are generated).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Short names for each mask (same length as masks). Default: file stems.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory (pairwise/ subdirectory for per-pair artifacts).",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--extended-aggregates",
        type=str,
        default="both",
        choices=("none", "param_bucket", "decoder_layer", "both"),
        help="Extra Jaccard sections in each JSON (attn/mlp/norm/other and/or decoder index).",
    )
    parser.add_argument(
        "--run-cka",
        action="store_true",
        help="Run mask_to_cka.py per pair (GPU + HF model; slow).",
    )
    parser.add_argument(
        "--cka-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model for CKA (must match mask architecture).",
    )
    parser.add_argument(
        "--cka-dataset",
        type=str,
        default="tulu3",
        help="Calibration HF id or registry key for mask_to_cka (DPO-style rows).",
    )
    parser.add_argument("--cka-device", type=str, default="cuda")
    parser.add_argument("--cka-n-samples", type=int, default=64)
    parser.add_argument("--cka-batch-size", type=int, default=4)
    parser.add_argument(
        "--skip-effective-rank",
        action="store_true",
        help="Pass --skip_effective_rank to export_layer_metrics_csv.py.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Write jaccard_matrix.png (requires matplotlib).",
    )
    parser.add_argument(
        "--smoke-debug",
        action="store_true",
        help="Generate two random masks from --smoke-reference (different seeds), then run the "
        "full suite with extended Jaccard, heatmap, effective rank, and CKA (unless "
        "--smoke-no-cka). Ignores positional mask paths.",
    )
    parser.add_argument(
        "--smoke-reference",
        type=str,
        default=None,
        help="Reference .pt (topology + sparsity behavior); required with --smoke-debug.",
    )
    parser.add_argument(
        "--smoke-seed-a",
        type=int,
        default=10001,
        help="RNG seed for first random mask (smoke mode).",
    )
    parser.add_argument(
        "--smoke-seed-b",
        type=int,
        default=20002,
        help="RNG seed for second random mask (smoke mode).",
    )
    parser.add_argument(
        "--smoke-sparsity",
        type=float,
        default=97.5,
        help="Target sparsity %% for both smoke masks.",
    )
    parser.add_argument(
        "--smoke-no-cka",
        action="store_true",
        help="In smoke mode, skip mask_to_cka (CPU-friendly dry run).",
    )
    parser.add_argument(
        "--probe-reports",
        action="store_true",
        help="After pairwise metrics, run linear probes (CAV-style) per mask; writes probe_reports/*.json (GPU).",
    )
    parser.add_argument(
        "--probe-mode",
        type=str,
        default="grpo",
        choices=["grpo", "dpo"],
        help="Calibration text layout for mask_probe_report.py.",
    )
    parser.add_argument(
        "--probe-dataset",
        type=str,
        default=None,
        help="Override HF dataset for probes (default: OpenR1-Math-220k for grpo, pipeline DPO id for dpo).",
    )
    parser.add_argument(
        "--probe-device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for per-mask probe runs (usually cuda).",
    )
    parser.add_argument("--probe-n-samples", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--probe-max-length", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cka = bool(args.run_cka)
    heatmap = bool(args.heatmap)
    extended = args.extended_aggregates
    skip_er = bool(args.skip_effective_rank)

    if args.smoke_debug:
        if not args.smoke_reference or not os.path.isfile(args.smoke_reference):
            print(
                "--smoke-debug requires an existing --smoke-reference .pt file.",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.masks:
            print(
                "[smoke] ignoring positional mask paths (--smoke-debug generates its own).",
                file=sys.stderr,
            )
        try:
            masks, labels = _materialize_smoke_masks(
                args.smoke_reference,
                out_dir,
                args.smoke_sparsity,
                args.smoke_seed_a,
                args.smoke_seed_b,
            )
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(2)
        run_cka = not args.smoke_no_cka
        heatmap = True
        extended = "both"
        skip_er = False
        print(
            f"[smoke] bells: extended={extended} heatmap={heatmap} "
            f"effective_rank=on run_cka={run_cka}",
            flush=True,
        )
    else:
        if len(args.masks) < 2:
            print("Need at least two mask paths, or use --smoke-debug.", file=sys.stderr)
            sys.exit(2)
        masks = [os.path.abspath(p) for p in args.masks]
        for p in masks:
            if not os.path.isfile(p):
                print(f"Missing mask file: {p}", file=sys.stderr)
                sys.exit(1)
        if args.labels is not None and len(args.labels) != len(masks):
            print("--labels must match number of masks", file=sys.stderr)
            sys.exit(1)
        labels = [
            _safe_label(masks[i], args.labels[i] if args.labels else None)
            for i in range(len(masks))
        ]

    pair_dir = out_dir / "pairwise"
    pair_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    rollup_rows: List[Dict[str, Any]] = []
    n = len(masks)
    jaccard_matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for i, j in combinations(range(n), 2):
        la, lb = labels[i], labels[j]
        tag = f"{la}__vs__{lb}"
        jpath = pair_dir / f"jaccard_{tag}.json"
        rep = _write_jaccard_json(
            masks[i],
            masks[j],
            jpath,
            device=args.device,
            extended=extended,
        )
        agg = float(rep["jaccard"]["aggregate"])
        jaccard_matrix[i][j] = jaccard_matrix[j][i] = agg

        cka_path: Optional[Path] = None
        if run_cka:
            cka_path = pair_dir / f"cka_{tag}.json"
            ok = _run_cka_pair(
                py,
                masks[i],
                masks[j],
                cka_path,
                model_name=args.cka_model,
                dataset_name=args.cka_dataset,
                device=args.cka_device,
                n_samples=args.cka_n_samples,
                batch_size=args.cka_batch_size,
            )
            if not ok:
                print(f"CKA failed for {tag}; continuing without CKA JSON", file=sys.stderr)
                cka_path = None

        csv_path = pair_dir / f"layer_metrics_{tag}.csv"
        _run_export_csv(
            py,
            masks[i],
            masks[j],
            csv_path,
            jpath,
            cka_path,
            skip_er,
        )

        row = {
            "mask_a": masks[i],
            "mask_b": masks[j],
            "label_a": la,
            "label_b": lb,
            "jaccard_aggregate": agg,
            "jaccard_mean": rep["jaccard"]["mean"],
            "jaccard_json": str(jpath),
            "layer_metrics_csv": str(csv_path),
        }
        if cka_path and cka_path.is_file():
            row["cka_json"] = str(cka_path)
        if "jaccard_by_param_bucket" in rep:
            for bucket, block in rep["jaccard_by_param_bucket"].items():
                row[f"jaccard_agg_{bucket}"] = block.get("aggregate_jaccard")
        rollup_rows.append(row)

    summary = {
        "labels": labels,
        "mask_paths": masks,
        "jaccard_matrix": jaccard_matrix,
        "pairwise": rollup_rows,
    }
    if args.smoke_debug:
        summary["smoke_debug"] = {
            "reference": os.path.abspath(args.smoke_reference),
            "seed_a": args.smoke_seed_a,
            "seed_b": args.smoke_seed_b,
            "sparsity_percent": args.smoke_sparsity,
            "run_cka": run_cka,
        }
    with open(out_dir / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({k for r in rollup_rows for k in r.keys()})
    with open(out_dir / "suite_pairwise.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rollup_rows:
            w.writerow(r)

    print(f"Wrote {out_dir / 'suite_summary.json'} and suite_pairwise.csv", flush=True)

    if heatmap:
        _maybe_heatmap(labels, jaccard_matrix, out_dir / "jaccard_matrix.png")

    if args.smoke_debug:
        _run_plot_layer_metrics(pair_dir, out_dir / "plots", py)

    if args.probe_reports:
        _run_probe_reports_for_masks(
            py,
            masks,
            labels,
            out_dir,
            model_name=args.cka_model,
            mode=args.probe_mode,
            dataset_name=args.probe_dataset,
            device=args.probe_device,
            n_samples=args.probe_n_samples,
            batch_size=args.probe_batch_size,
            max_length=args.probe_max_length,
        )


def _run_probe_reports_for_masks(
    py: str,
    mask_paths: List[str],
    labels: List[str],
    out_dir: Path,
    *,
    model_name: str,
    mode: str,
    dataset_name: Optional[str],
    device: str,
    n_samples: int,
    batch_size: int,
    max_length: int,
) -> None:
    probe_dir = out_dir / "probe_reports"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_py = _REPO_ROOT / "src/cold_start/mask_probe_report.py"
    per_mask: Dict[str, Any] = {}
    for mp, lb in zip(mask_paths, labels):
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in lb)
        outj = probe_dir / f"{safe}_probe_report.json"
        cmd: List[str] = [
            py,
            str(probe_py),
            mp,
            "-o",
            str(outj),
            "--model-name",
            model_name,
            "--mode",
            mode,
            "--device",
            device,
            "--n-samples",
            str(n_samples),
            "--batch-size",
            str(batch_size),
            "--max-length",
            str(max_length),
        ]
        if dataset_name:
            cmd.extend(["--dataset-name", dataset_name])
        print("RUN:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if r.returncode != 0:
            print(f"[probe] FAILED for {lb} ({mp})", file=sys.stderr)
            continue
        try:
            with open(outj, "r", encoding="utf-8") as f:
                data = json.load(f)
            per_mask[lb] = {
                "path": mp,
                "json": str(outj),
                "summary": data.get("summary"),
            }
        except OSError as e:
            per_mask[lb] = {"path": mp, "error": repr(e)}

    summary_path = out_dir / "suite_summary.json"
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as f:
            suite = json.load(f)
        suite["per_mask_linear_probes"] = per_mask
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(suite, f, indent=2, ensure_ascii=False)
    with open(out_dir / "suite_probe_index.json", "w", encoding="utf-8") as f:
        json.dump(per_mask, f, indent=2, ensure_ascii=False)
    print(f"[probe] Wrote probe reports under {probe_dir}", flush=True)


if __name__ == "__main__":
    main()

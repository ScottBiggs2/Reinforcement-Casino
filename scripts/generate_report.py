#!/usr/bin/env python3
"""
Generate speedup comparison report from benchmark timing results.

Usage:
    python scripts/generate_report.py /path/to/benchmark_output_dir [/path/to/ablation_output_dir]

Reads timing_results.json from each subdirectory and prints a comparison table.
"""

import json
import glob
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(output_dir):
    results = []
    for f in sorted(glob.glob(os.path.join(output_dir, "*/timing_results.json"))):
        with open(f) as fh:
            d = json.load(fh)
            d["_source"] = os.path.basename(os.path.dirname(f))
            results.append(d)
    return results


def print_speedup_report(results):
    if not results:
        print("No results found.")
        return

    print()
    print("=" * 80)
    print("SPARSE TRAINING SPEEDUP REPORT")
    print("=" * 80)

    # Find dense AdamW as baseline
    baseline = None
    for r in results:
        if r.get("method") == "dense" and r.get("optimizer") == "adamw":
            baseline = r
            break

    # Header
    print()
    print(f"{'Method':<20} {'Optimizer':<15} {'Wall(s)':<10} {'Step(s)':<10} {'GPU(s)':<10} {'GPU/Step':<10} {'Speedup':<10} {'Extra'}")
    print("-" * 105)

    for r in results:
        method = r.get("method", "?")
        opt = r.get("optimizer", "?")
        wall = r.get("wall_time", 0)
        step_wall = r.get("time_per_step_wall", 0)
        gpu = r.get("gpu_time", 0)
        gpu_step = r.get("time_per_step_gpu", 0)

        extra_parts = []
        if "block_size_bsr" in r:
            extra_parts.append(f"bsr={r['block_size_bsr']}")
        if "block_size_adam" in r:
            extra_parts.append(f"adam={r['block_size_adam']}")
        if "lora_rank" in r:
            extra_parts.append(f"r={r['lora_rank']}")
        extra = " ".join(extra_parts)

        # Speedup relative to dense AdamW (by gpu time per step)
        if baseline and gpu_step > 0:
            baseline_gpu_step = baseline.get("time_per_step_gpu", 0)
            speedup = f"{baseline_gpu_step / gpu_step:.2f}x" if baseline_gpu_step > 0 else "-"
        else:
            speedup = "baseline" if method == "dense" and opt == "adamw" else "-"

        print(f"{method:<20} {opt:<15} {wall:<10.1f} {step_wall:<10.2f} {gpu:<10.1f} {gpu_step:<10.2f} {speedup:<10} {extra}")

    print()
    if baseline:
        print(f"Baseline: dense + adamw (GPU time/step = {baseline.get('time_per_step_gpu', 0):.2f}s)")
    print()


def print_ablation_report(results):
    if not results:
        return

    print()
    print("=" * 80)
    print("BLOCK SIZE ABLATION REPORT")
    print("=" * 80)
    print()
    print(f"{'BSR':<8} {'Adam':<8} {'Wall(s)':<10} {'Step(s)':<10} {'GPU(s)':<10} {'GPU/Step':<10}")
    print("-" * 56)

    # Sort by block_size_bsr then block_size_adam
    results.sort(key=lambda r: (r.get("block_size_bsr", 0), r.get("block_size_adam", 0)))

    for r in results:
        bsr = r.get("block_size_bsr", "?")
        adam = r.get("block_size_adam", "?")
        wall = r.get("wall_time", 0)
        step_wall = r.get("time_per_step_wall", 0)
        gpu = r.get("gpu_time", 0)
        gpu_step = r.get("time_per_step_gpu", 0)
        print(f"{bsr:<8} {adam:<8} {wall:<10.1f} {step_wall:<10.2f} {gpu:<10.1f} {gpu_step:<10.2f}")

    # Find best config
    best = min(results, key=lambda r: r.get("time_per_step_gpu", float("inf")))
    print()
    print(f"Best: bsr={best.get('block_size_bsr')} adam={best.get('block_size_adam')} "
          f"(GPU/step = {best.get('time_per_step_gpu', 0):.2f}s)")
    print()


def plot_speedup_chart(results, out_dir):
    """Bar chart comparing GPU time/step across methods."""
    labels = []
    gpu_steps = []
    colors = []
    color_map = {
        ("dense", "adamw"): "#4C72B0",
        ("dense", "sgd"): "#55A868",
        ("lora", "adamw"): "#C44E52",
        ("sparse_bsr", "sparse_adamw"): "#8172B2",
    }

    for r in results:
        method = r.get("method", "?")
        opt = r.get("optimizer", "?")
        gpu_step = r.get("time_per_step_gpu", r.get("time_per_step_wall", 0))

        label = f"{method}\n{opt}"
        if "lora_rank" in r:
            label += f"\nr={r['lora_rank']}"
        if "block_size_bsr" in r:
            label += f"\nbsr={r['block_size_bsr']}"

        labels.append(label)
        gpu_steps.append(gpu_step)
        colors.append(color_map.get((method, opt), "#CCCCCC"))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, gpu_steps, color=colors, width=0.6, edgecolor="white", linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, gpu_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Baseline line
    baseline_val = None
    for r in results:
        if r.get("method") == "dense" and r.get("optimizer") == "adamw":
            baseline_val = r.get("time_per_step_gpu", r.get("time_per_step_wall", 0))
            break
    if baseline_val:
        ax.axhline(y=baseline_val, color="#4C72B0", linestyle="--", alpha=0.5, label=f"Dense AdamW baseline ({baseline_val:.2f}s)")
        ax.legend(fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("GPU Time per Step (s)", fontsize=12)
    ax.set_title("Training Method Speedup Comparison (Single H200)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(gpu_steps) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "speedup_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


def plot_ablation_chart(results, out_dir):
    """Grouped bar chart for block size ablation."""
    results.sort(key=lambda r: (r.get("block_size_bsr", 0), r.get("block_size_adam", 0)))

    labels = []
    gpu_steps = []
    for r in results:
        bsr = r.get("block_size_bsr", "?")
        adam = r.get("block_size_adam", "?")
        gpu_step = r.get("time_per_step_gpu", r.get("time_per_step_wall", 0))
        labels.append(f"bsr={bsr}\nadam={adam}")
        gpu_steps.append(gpu_step)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    bars = ax.bar(x, gpu_steps, color=colors, width=0.6, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, gpu_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Highlight best
    best_idx = gpu_steps.index(min(gpu_steps))
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("GPU Time per Step (s)", fontsize=12)
    ax.set_title("Block Size Ablation (Single H200)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(gpu_steps) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "ablation_block_size.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_report.py <benchmark_dir> [ablation_dir]")
        print()
        print("Auto-detect: python scripts/generate_report.py auto")
        sys.exit(1)

    if sys.argv[1] == "auto":
        # Find most recent benchmark and ablation dirs
        scratch_user_root = os.environ.get("SCRATCH_USER_ROOT") or f"/scratch/{os.environ.get('USER', 'unknown')}"
        base = os.environ.get("RL_CASINO_OUTPUTS_DIR") or os.path.join(scratch_user_root, "rl_casino_outputs")
        bench_dirs = sorted(glob.glob(os.path.join(base, "benchmark_*")))
        ablation_dirs = sorted(glob.glob(os.path.join(base, "ablation_blocksize_*")))

        if bench_dirs:
            print(f"Found benchmark dir: {bench_dirs[-1]}")
            bench_results = load_results(bench_dirs[-1])
            print_speedup_report(bench_results)

            # Save
            summary = os.path.join(bench_dirs[-1], "benchmark_summary.json")
            with open(summary, "w") as f:
                json.dump(bench_results, f, indent=2)
            print(f"Summary saved to {summary}")
        else:
            print("No benchmark directories found.")

        if ablation_dirs:
            print(f"Found ablation dir: {ablation_dirs[-1]}")
            ablation_results = load_results(ablation_dirs[-1])
            # Filter to only sparse_bsr method
            ablation_results = [r for r in ablation_results if r.get("method") == "sparse_bsr"]
            print_ablation_report(ablation_results)

            summary = os.path.join(ablation_dirs[-1], "ablation_summary.json")
            with open(summary, "w") as f:
                json.dump(ablation_results, f, indent=2)
            print(f"Summary saved to {summary}")
        else:
            print("No ablation directories found.")
            ablation_results = []

        # Generate charts
        out_dir = bench_dirs[-1] if bench_dirs else (ablation_dirs[-1] if ablation_dirs else ".")
        if bench_results:
            plot_speedup_chart(bench_results, out_dir)
        if ablation_results:
            plot_ablation_chart(ablation_results, out_dir)
    else:
        bench_dir = sys.argv[1]
        bench_results = load_results(bench_dir)
        print_speedup_report(bench_results)

        if len(sys.argv) > 2:
            ablation_dir = sys.argv[2]
            ablation_results = load_results(ablation_dir)
            ablation_results = [r for r in ablation_results if r.get("method") == "sparse_bsr"]
            print_ablation_report(ablation_results)


if __name__ == "__main__":
    main()

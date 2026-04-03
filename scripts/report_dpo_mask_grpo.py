#!/usr/bin/env python3
"""
Generate comparison report for DPO-mask GRPO experiments.
Pulls metrics from wandb and creates summary table + plots.

Usage:
    python scripts/report_dpo_mask_grpo.py
    python scripts/report_dpo_mask_grpo.py --run_filter "dpo_mask_grpo"
"""

import argparse
import json
import os

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def fetch_runs(project: str, run_filter: str) -> list:
    """Fetch matching runs from wandb."""
    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": {"$regex": run_filter}})
    print(f"Found {len(runs)} runs matching '{run_filter}'")
    return runs


def extract_metrics(run) -> dict:
    """Extract key metrics from a wandb run."""
    history = run.history(keys=[
        "train/rewards/accuracy_reward/mean",
        "train/rewards/format_number_reward/mean",
        "train/rewards/format_reasoning_reward/mean",
        "train/loss",
        "train/grad_norm",
    ], samples=10000)

    n_steps = len(history)
    if n_steps == 0:
        return None

    # Use last 25% of steps for final metrics (more stable)
    tail_start = int(n_steps * 0.75)
    tail = history.iloc[tail_start:]

    acc_col = "train/rewards/accuracy_reward/mean"
    fmt_num_col = "train/rewards/format_number_reward/mean"
    fmt_reason_col = "train/rewards/format_reasoning_reward/mean"
    loss_col = "train/loss"

    return {
        "run_name": run.name,
        "status": run.state,
        "total_steps": n_steps,
        # Overall means
        "accuracy_reward_mean": history[acc_col].mean() if acc_col in history else None,
        "accuracy_reward_std": history[acc_col].std() if acc_col in history else None,
        # Last 25% means (converged performance)
        "accuracy_final": tail[acc_col].mean() if acc_col in tail else None,
        "format_number_final": tail[fmt_num_col].mean() if fmt_num_col in tail else None,
        "format_reasoning_final": tail[fmt_reason_col].mean() if fmt_reason_col in tail else None,
        "loss_final": tail[loss_col].mean() if loss_col in tail else None,
        # Raw history for plotting
        "_history": history,
    }


def print_summary_table(results: list[dict]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("DPO Mask → GRPO Transfer Experiment Summary")
    print("=" * 80)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    df = df.sort_values("accuracy_final", ascending=False)

    print(f"\n{'Run':<45} {'Steps':>6} {'Acc(final)':>11} {'Acc(all)':>11} {'Loss(final)':>12}")
    print("-" * 90)
    for _, row in df.iterrows():
        name = row["run_name"][:44]
        print(f"{name:<45} {row['total_steps']:>6} "
              f"{row['accuracy_final']:>11.4f} {row['accuracy_reward_mean']:>11.4f} "
              f"{row['loss_final']:>12.6f}")

    print("\n" + "-" * 90)
    return df


def plot_comparison(results: list[dict], output_dir: str):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("DPO Mask → GRPO Transfer: Comparison", fontsize=14, fontweight="bold")

    metrics = [
        ("train/rewards/accuracy_reward/mean", "Accuracy Reward", axes[0, 0]),
        ("train/rewards/format_number_reward/mean", "Format Number Reward", axes[0, 1]),
        ("train/rewards/format_reasoning_reward/mean", "Format Reasoning Reward", axes[1, 0]),
        ("train/loss", "Training Loss", axes[1, 1]),
    ]

    for metric_key, title, ax in metrics:
        for r in results:
            history = r["_history"]
            if metric_key not in history.columns:
                continue
            values = history[metric_key].values
            # Smooth with rolling window for readability
            window = max(1, len(values) // 20)
            smoothed = pd.Series(values).rolling(window=window, min_periods=1).mean()
            label = r["run_name"].replace("dpo_mask_grpo_", "").split("_202")[0]
            ax.plot(smoothed, label=label, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dpo_mask_grpo_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    # Bar chart for final accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    names = []
    accs = []
    for r in sorted(results, key=lambda x: x["accuracy_final"] or 0, reverse=True):
        label = r["run_name"].replace("dpo_mask_grpo_", "").split("_202")[0]
        names.append(label)
        accs.append(r["accuracy_final"] or 0)

    colors = ["#2ecc71" if "baseline" not in n else "#3498db" for n in names]
    ax.bar(names, accs, color=colors)
    ax.set_title("Final Accuracy Reward (last 25% of training)")
    ax.set_ylabel("Accuracy Reward Mean")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)

    bar_path = os.path.join(output_dir, "dpo_mask_grpo_accuracy_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    print(f"Bar chart saved to: {bar_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="huggingface",
                        help="wandb project name")
    parser.add_argument("--run_filter", type=str, default="dpo_mask_grpo",
                        help="Regex filter for run names")
    parser.add_argument("--output_dir", type=str, default="mask_swapping/report",
                        help="Output directory for plots and summary")
    args = parser.parse_args()

    runs = fetch_runs(args.project, args.run_filter)
    if not runs:
        print("No runs found. Check project name and filter.")
        return

    results = []
    for run in runs:
        print(f"  Fetching: {run.name} ({run.state})")
        metrics = extract_metrics(run)
        if metrics:
            results.append(metrics)

    if not results:
        print("No valid results to report.")
        return

    df = print_summary_table(results)
    plot_comparison(results, args.output_dir)

    # Save raw summary as CSV
    csv_path = os.path.join(args.output_dir, "dpo_mask_grpo_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

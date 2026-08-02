#!/usr/bin/env python
"""Rebuild the 4-arm DPO loss figure (Light-R1 + Tulu3 panels) from wandb.

Pulls train/loss vs train/global_step for the 2026-07-31 rebuttal rerun arms in
entity xxiellan-northeastern-university, project huggingface, and renders the
two panels in the same style as the submitted figure (dense blue solid,
magnitude red dashed, oracle green dotted, random gray dash-dot).

Run after all eight arms finish:
    python scripts/plot_reb_dpo_4arm_loss.py
Optional: --project/--entity overrides, --out prefix.
"""
import argparse

import matplotlib.pyplot as plt
import wandb

# After the 2026-07-31 reorg, runs live in per-dataset projects.
PANEL_PROJECT = {"DPO Light R1": "dpo_lightr1", "DPO Tulu3": "dpo_tulu3"}

# label -> (run_name, color, linestyle)
PANELS = {
    "DPO Light R1": {
        "DPO Dense Training": ("dense_dpo_lightr1_500steps_b200", "C0", "-"),
        "DPO Magnitude Rewinding Step200": (
            "sparse_dpo_lightr1_reb_magnitude_step200_500steps", "red", "--"),
        # oracle reuses the rho-sweep run (duplicate B200 rerun cancelled 07-31)
        "DPO Oracle Light R1": (
            "sparse_dpo_lightr1_oracle_step500_sp97.5_500steps", "green", ":",
            "rl_casino_transfer_v1"),
        "DPO Random Mask": (
            "sparse_dpo_lightr1_random_sp97.5_seed42_500steps_b200", "gray", "-."),
    },
    "DPO Tulu3": {
        "DPO Dense Training": ("dense_dpo_tulu3_500steps_b200", "C0", "-"),
        "DPO Magnitude Rewinding Step200": (
            "sparse_dpo_tulu3_reb_magnitude_step200_500steps", "red", "--"),
        # relaunched on B200 (job 245164): the sweep run a0gb8auf is Explorer H200
        "DPO Oracle Tulu3": (
            "sparse_dpo_tulu3_oracle_step500_sp97.5_500steps_b200", "green", ":"),
        "DPO Random Mask": (
            "sparse_dpo_tulu3_random_sp97.5_seed42_500steps_b200", "gray", "-."),
    },
}


def fetch(api, entity, project, run_name):
    runs = list(api.runs(f"{entity}/{project}", {"display_name": run_name}))
    if not runs:  # fallback: run not yet moved out of the default project
        runs = list(api.runs(f"{entity}/huggingface", {"display_name": run_name}))
    if not runs:
        raise SystemExit(f"missing run in {project} (and huggingface): {run_name}")
    def _step(r):
        try:
            v = r.summary.get("train/global_step")
            return v if isinstance(v, (int, float)) else 0
        except Exception:  # wandb.old.summary decode failure on malformed summaries
            return 0
    run = max(runs, key=_step)
    hist = run.history(keys=["train/global_step", "train/loss"], pandas=True, samples=2000)
    hist = hist.dropna().sort_values("train/global_step")
    return hist["train/global_step"], hist["train/loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="xxiellan-northeastern-university")
    ap.add_argument("--out", default="docs/paper_drafts/reb_dpo_loss_4arm")
    args = ap.parse_args()

    api = wandb.Api()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (panel, arms) in zip(axes, PANELS.items()):
        for label, arm in arms.items():
            run_name, color, ls = arm[0], arm[1], arm[2]
            proj = arm[3] if len(arm) > 3 else PANEL_PROJECT[panel]
            x, y = fetch(api, args.entity, proj, run_name)
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.6, label=label)
        ax.set_title(f"{panel} Loss over Time", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_xlim(0, 500)
        ax.legend(fontsize=8)
    axes[1].set_ylim(0.60, 0.81)  # match the submitted Tulu3 panel's clipped range
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()

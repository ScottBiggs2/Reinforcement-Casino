"""Qwen3 3-arm DPO loss figure (8B + 32B panels), uniform colours.

Same panel styling and arm colours as plot_reb_dpo_4arm_uniform.py (random
orange, dense blue, oracle green; solid raw curves). Runs are the qwen3
registry arms (FIGURE_RUN_REGISTRY.md "Qwen3 DPO 图"), pinned by run ID.
The 32B dense curve is the backfilled full 1-500 run 40yoz8rd, NOT the
partial live run 97idxxzx.

A None run ID is resolved by exact run name in the qwen3 project (for arms
whose training is still in flight) — backfill the ID into the registry once
the run finishes.

Usage: python scripts/plot_qwen3_3arm_uniform.py [--out prefix] [--metric train/loss]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT = "xxiellan-northeastern-university"
PROJ = "qwen3"
INK, GRIDC, SPINEC = "#000000", "#d9d9d9", "#cccccc"

ORANGE, BLUE, GREEN = "#ff7f0e", "#2b7eb8", "#2b9d32"

ARMS = [
    ("DPO Random Mask", ORANGE),
    ("DPO Dense Training", BLUE),
    ("DPO Oracle", GREEN),
]

# panel -> (run_id or None, fallback exact run name) in ARMS order
PANELS = {
    "Qwen3-8B Light R1": [
        ("7umzk6t8", "sparse_dpo_qwen3_8b_light_r1_random_sp97.5_seed42_500steps"),
        ("8pl32d0c", "dense_dpo_qwen3_8b_light_r1_scott_full"),
        ("t4sclw40", "sparse_dpo_qwen3_8b_light_r1_oracle_step500_500steps"),
    ],
    "Qwen3-32B Light R1": [
        (None, "sparse_dpo_qwen3_32b_light_r1_random_sp97.5_seed42_500steps"),
        ("40yoz8rd", "dense_dpo_qwen3_32b_light_r1_scott_full"),
        ("gg5y7yre", "sparse_dpo_qwen3_32b_light_r1_oracle_step500_500steps"),
    ],
}

LEGEND_LOC = {"Qwen3-8B Light R1": "center right", "Qwen3-32B Light R1": "center right"}


def resolve(api, run_id, name):
    if run_id:
        return api.run(f"{ENT}/{PROJ}/{run_id}")
    matches = [r for r in api.runs(f"{ENT}/{PROJ}") if r.name == name]
    if not matches:
        raise SystemExit(f"no run id pinned and no run named {name!r} in {PROJ} — "
                         "is the training up yet?")
    if len(matches) > 1:
        raise SystemExit(f"{len(matches)} runs named {name!r} — pin the ID in "
                         "FIGURE_RUN_REGISTRY.md and in PANELS above")
    print(f"[resolve] {name} -> {matches[0].id}  (backfill this ID into the registry)")
    return matches[0]


def fetch(api, run_id, name, metric):
    run = resolve(api, run_id, name)
    hist = run.history(keys=["train/global_step", metric], pandas=True, samples=2000)
    hist = hist.dropna().sort_values("train/global_step")
    return hist["train/global_step"], hist[metric]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper_drafts/qwen3_3arm_uniform")
    ap.add_argument("--metric", default="train/loss")
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip arms whose run is not up yet (progress snapshots)")
    args = ap.parse_args()
    metric_label = args.metric.split("/", 1)[-1].replace("_", " ").title()

    api = wandb.Api()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), facecolor="white")
    for ax, (panel, runs) in zip(axes, PANELS.items()):
        for (label, colour), (run_id, name) in zip(ARMS, runs):
            try:
                x, y = fetch(api, run_id, name, args.metric)
            except SystemExit:
                if args.allow_missing:
                    print(f"[skip] {name}")
                    continue
                raise
            ax.plot(x, y, color=colour, linewidth=1.6, label=label)
        ax.set_title(f"{panel} {metric_label} over Time", fontsize=11,
                     fontweight="bold", color=INK, pad=8)
        ax.set_xlabel("Step", fontsize=10, color=INK)
        ax.set_ylabel(metric_label, fontsize=10, color=INK)
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.grid(True, color=GRIDC, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(SPINEC)
            s.set_linewidth(0.8)
        ax.tick_params(colors=INK, labelsize=9)
        leg = ax.legend(loc=LEGEND_LOC[panel], fontsize=8.5, frameon=True,
                        edgecolor=SPINEC, framealpha=1.0, borderpad=0.5)
        leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor="white")
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()

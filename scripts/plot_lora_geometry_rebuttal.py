"""Three LoRA-rebuttal figures from data already in the repo (no cluster access needed).

Data sources (pulled off Explorer 2026-07-31, now tracked in-repo):
  docs/run_logs/delta_analysis_2026-07-25/delta_*.json   (job 8725606, arms 0-2)
  docs/run_logs/pref_eval_2026-07-27/*.json              (jobs 8727254, 8763502, 8787043)

Figure 1 — stable-rank spectrum. Per-matrix stable rank ||dW||_F^2/||dW||_2^2 of the
  dense update, sorted, one line per checkpoint, against the r=64 cap that LoRA imposes
  by construction at the matched trainable budget. The measured tensors are the FIRST 48
  matched 2-D params (embed + layers 0-6) — rank_max_tensors=48 in job 8725606 — so this
  is an early-layer sample, not full depth. The AICR rerun lifts the cap.

Figure 2 — matched-budget energy capture, the "geometry" metric. For each matrix, two
  numbers on one axis:
    * what our prior captures: fraction of ||dW||^2 in its top-2.5% coordinates
      (measured, per_layer top_2.5pct);
    * what ANY rank-64 update can capture: best rank-r Frobenius capture is
      sum_{i<=r} sigma_i^2 <= r*sigma_1^2 = (r/srank)*||dW||_F^2  (Eckart-Young),
      an upper bound that needs no SVD — it is exact only if the top r singular
      values are all equal, so the true LoRA capture is LOWER.
  embed_tokens is excluded: the LoRA baseline (r=64, all-linear) does not adapt it.

Figure 3 — held-out margin vs policy displacement (Tulu3 tail, cross-dataset, n=500).
  x is -r_chosen = beta*(log pi_ref - log pi) on chosen responses: how far the policy
  moved AWAY from the responses it is supposed to prefer. LoRA's 3x margin at 20x LR
  rides on 3.1x displacement while accuracy moves -0.004 — the Razin et al. (ICLR 2025)
  failure mode, measured on our own arms. Sparse arms sit an order of magnitude closer
  to the reference at comparable accuracy movement per unit margin.

Usage:  python scripts/plot_lora_geometry_rebuttal.py  [--outdir docs/paper_drafts]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LORA_R = 64

CKPTS = [  # tag, display label, color
    ("llama_dpo_tulu3", "DPO · Tülu-3 (Llama-8B)", BLUE),
    ("llama_grpo_math220k", "GRPO · Math-220k (Llama-8B)", ORANGE),
    ("qwen3_dpo_lightr1", "DPO · Light-R1 (Qwen3-8B)", AQUA),
]


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)


def load_delta(datadir, tag):
    return json.load(open(os.path.join(datadir, f"delta_{tag}.json")))


def fig_stable_rank(datadir, out):
    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor="white")
    for tag, label, color in CKPTS:
        d = load_delta(datadir, tag)
        sr = sorted(r["stable_rank"] for r in d["stable_ranks"])
        n = len(sr)
        above = sum(s > LORA_R for s in sr)
        x = (np.arange(n) + 0.5) / n * 100
        med = sr[n // 2]
        ax.plot(x, sr, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{label} — median {med:.0f}, {above}/{n} above the cap",
                zorder=3)
    ax.axhline(LORA_R, color=INK, linewidth=1.6, linestyle=(0, (5, 2)))
    # right half below the line is empty (all three curves are >128 there)
    ax.text(99, LORA_R * 0.93, f"LoRA r={LORA_R} — the highest stable rank any\n"
            "adapter update can have, at the matched 2% budget",
            fontsize=10.5, color=INK, va="top", ha="right", linespacing=1.35)
    ax.set_yscale("log")
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_yticklabels(["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"])
    ax.set_title("The dense RL update is not low-rank\n"
                 "stable rank of ΔW per matrix, first 48 matched 2-D params",
                 fontsize=15, fontweight="700", color=INK, pad=12)
    ax.set_xlabel("matrices, sorted by stable rank  (percentile)", fontsize=13, color=MUTED)
    ax.set_ylabel("stable rank  ‖ΔW‖²_F / ‖ΔW‖²₂", fontsize=13, color=MUTED)
    ax.legend(loc="upper left", fontsize=10.5, frameon=False)
    style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white")
    print(f"wrote {out}.png/.pdf")


def fig_energy_capture(datadir, specdir, out):
    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor="white")

    # analytic ceiling for ANY rank-64 update, as a function of stable rank
    xs = np.geomspace(4, 1500, 400)
    ceil = np.minimum(1.0, LORA_R / xs)
    ax.plot(xs, ceil, color=INK, linewidth=2, linestyle=(0, (5, 2)), zorder=2)
    ax.fill_between(xs, ceil, 1.02, color=GRID, alpha=0.55, zorder=1)
    ax.text(430, 0.62, "unreachable by any\nrank-64 update",
            fontsize=11, color=MUTED, ha="center")

    # all three checkpoints are Llama-8B: Light-R1 comes from the full-depth
    # AICR 243173 spectrum run (the LoRA arm's task), not the Qwen3 delta
    files = [
        (os.path.join(datadir, "delta_llama_dpo_tulu3.json"), "DPO · Tülu-3", BLUE),
        (os.path.join(datadir, "delta_llama_grpo_math220k.json"), "GRPO · Math-220k", ORANGE),
        (os.path.join(specdir, "delta_llama_dpo_lightr1_spec.json"), "DPO · Light-R1", AQUA),
    ]
    printed = []
    for path, label, color in files:
        d = json.load(open(path))
        conc = {p["name"]: p for p in d["per_layer"]}
        pts = [(r["stable_rank"], min(1.0, conc[r["name"]]["top_2.5pct"]))
               for r in d["stable_ranks"]
               if "embed_tokens" not in r["name"] and "lm_head" not in r["name"]
               and r["name"] in conc]
        sx, sy = zip(*pts)
        ax.scatter(sx, sy, s=42, color=color, edgecolors="white", linewidths=1.2,
                   label=label, zorder=4)
        lora_bound = [min(1.0, LORA_R / x) for x in sx]
        printed.append((label, np.median(sy), np.median(lora_bound)))

    ax.set_xscale("log")
    ax.set_xlim(4, 1500)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512", "1024"])
    ax.set_title("The RL update fits in 2.5% of weights, but not in rank 64\n"
                 "Llama-8B dense checkpoints",
                 fontsize=15, fontweight="700", color=INK, pad=12)
    ax.set_xlabel("stable rank of the dense update ΔW  (per matrix)", fontsize=13, color=MUTED)
    ax.set_ylabel("fraction of ‖ΔW‖² captured", fontsize=13, color=MUTED)
    ax.annotate("top-2.5% coordinates of each matrix\n(our prior — measured)",
                (7, 0.965), fontsize=11, color=INK, va="top")
    leg = ax.legend(loc="lower left", fontsize=10.5, frameon=True,
                    edgecolor=GRID, facecolor="white", title="dense checkpoint")
    leg.get_title().set_fontsize(10)
    style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white")
    print(f"wrote {out}.png/.pdf")
    for label, s_med, l_med in printed:
        print(f"  {label:34s} sparse capture median {s_med:.3f}   rank-64 ceiling median {l_med:.3f}")


SPEC_CKPTS = [  # tag, display label, color — full-depth exact-spectrum runs (AICR 243173)
    ("llama_dpo_lightr1", "DPO · Light-R1 (Llama-8B) — the LoRA arm's task", BLUE),
    ("llama_grpo_math220k", "GRPO · Math-220k (Llama-8B)", ORANGE),
]


def fig_energy_capture_exact(specdir, out):
    """Fig 2 upgraded: EXACT best-rank-64 capture (top-256 spectra, all 224 linear
    matrices) instead of the Eckart-Young bound. embed/lm_head excluded to match the
    r=64 all-linear LoRA baseline's adapted set."""
    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor="white")

    xs = np.geomspace(2, 700, 400)
    ax.plot(xs, np.minimum(1.0, LORA_R / xs), color=MUTED, linewidth=1.6,
            linestyle=(0, (5, 2)), zorder=2)
    ax.text(400, 0.31, "Eckart–Young ceiling\n≤ r / stable rank",
            fontsize=10, color=MUTED, ha="center", linespacing=1.3)

    meds = []
    for tag, label, color in SPEC_CKPTS:
        d = json.load(open(os.path.join(specdir, f"delta_{tag}_spec.json")))
        conc = {p["name"]: p for p in d["per_layer"]}
        rows = [r for r in d["stable_ranks"]
                if "embed_tokens" not in r["name"] and "lm_head" not in r["name"]]
        sr = [r["stable_rank"] for r in rows]
        sparse = [min(1.0, conc[r["name"]]["top_2.5pct"]) for r in rows]
        cap64 = [sum(r["sigma_sq_topk"][:LORA_R]) / r["fro_sq"] for r in rows]
        ax.scatter(sr, sparse, s=26, color=color, alpha=0.75, linewidths=0, zorder=4)
        ax.scatter(sr, cap64, s=30, facecolors="none", edgecolors=color,
                   linewidths=1.3, alpha=0.8, zorder=3)
        meds.append((label, np.median(sparse), np.median(cap64)))
        # legend proxy: one filled + one hollow marker per checkpoint
        ax.scatter([], [], s=42, color=color, label=label)
    ax.scatter([], [], s=42, color=INK, label="filled — top-2.5% coordinates (ours)")
    ax.scatter([], [], s=46, facecolors="none", edgecolors=INK, linewidths=1.3,
               label="hollow — best rank-64 approximation (exact)")

    ax.set_xscale("log")
    ax.set_xlim(2, 700)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256", "512"])
    ax.set_title("Same parameter budget: coordinates hold the update,\n"
                 "the best rank-64 approximation holds ~10% of it",
                 fontsize=15, fontweight="700", color=INK, pad=12)
    ax.set_xlabel("stable rank of the dense update ΔW  (per matrix, all 224 linear)",
                  fontsize=13, color=MUTED)
    ax.set_ylabel("fraction of ‖ΔW‖² captured", fontsize=13, color=MUTED)
    ax.legend(loc="center left", fontsize=10, frameon=True, edgecolor=GRID,
              facecolor="white")
    style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white")
    print(f"wrote {out}.png/.pdf")
    for label, s_med, c_med in meds:
        print(f"  {label:44s} sparse median {s_med:.3f}   exact rank-64 median {c_med:.3f}")


ARMS = [  # file, label, color, label position (data coords; None = beside the point)
    ("base_llama", "base (sanity)", MUTED, (-0.15, 0.185)),
    ("sparse_random_lightr1", "random mask 2.5%", MUTED, (-0.15, 0.225)),
    ("sparse_warmmag200_lightr1", "warm-mag 2.5%", BLUE, (-0.15, 0.265)),
    ("sparse_oracle_lightr1", "sparse oracle 2.5%", BLUE, (-0.15, 0.305)),
    ("lora_r64_lr5e-6", "LoRA r=64 @5e-6", ORANGE, None),
    ("lora_r64_lr1e-4", "LoRA r=64 @1e-4", ORANGE, None),
]


def fig_margin_displacement(evaldir, out):
    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor="white")
    rows = {}
    for f, label, color, pos in ARMS:
        d = json.load(open(os.path.join(evaldir, f"{f}.json")))
        x = -d["reward_chosen_mean"]
        y = d["reward_margin_mean"]
        lo, hi = d["reward_margin_ci95"]
        acc = d["preference_accuracy"]
        rows[f] = (x, y, acc)
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], color=color, marker="o",
                    markersize=8, capsize=4, capthick=1.5, elinewidth=1.5, zorder=3)
        text = f"{label} — acc {acc:.3f}" if f != "base_llama" else label
        if pos is None:  # LoRA points are isolated; label beside/below the point
            dx, dy, ha = ((-12, 4, "right") if x > 2 else (14, -4, "left"))
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=10, color=INK)
        else:  # bottom-left cluster: stack labels in the empty upper-left,
            # thin leader line down to each point
            ax.annotate(text, (x, y), xytext=pos, ha="left", va="center",
                        fontsize=10, color=INK,
                        arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1.1,
                                        shrinkB=7))

    # the 20x-LR arrow inside LoRA: margin x3, displacement x3.1, accuracy -0.004
    (x0, y0, a0), (x1, y1, a1) = rows["lora_r64_lr5e-6"], rows["lora_r64_lr1e-4"]
    ax.annotate("", (x1 - 0.06, y1 - 0.008), (x0 + 0.08, y0 + 0.004),
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.4,
                                linestyle=(0, (4, 3))))
    ax.text(2.85, 0.038,
            f"LoRA LR ×20:\nmargin ×{y1/y0:.1f},  displacement ×{x1/x0:.1f},\n"
            f"accuracy {a1-a0:+.3f}",
            fontsize=10.5, color=ORANGE, ha="center", linespacing=1.3)

    ax.set_title("Held-out margin is bought with policy displacement\n"
                 "Tülu-3 tail (cross-dataset), n=500 pairs, β=0.1",
                 fontsize=15, fontweight="700", color=INK, pad=12)
    ax.set_xlabel("policy displacement on chosen responses   −r̄_chosen = β·(log π_ref − log π)",
                  fontsize=12.5, color=MUTED)
    ax.set_ylabel("held-out reward margin", fontsize=13, color=MUTED)
    ax.set_xlim(-0.25, 4.0)
    style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white")
    print(f"wrote {out}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="docs/run_logs/delta_analysis_2026-07-25")
    ap.add_argument("--specdir", default="docs/run_logs/delta_spectrum_2026-07-31")
    ap.add_argument("--evaldir", default="docs/run_logs/pref_eval_2026-07-27")
    ap.add_argument("--outdir", default="docs/paper_drafts")
    args = ap.parse_args()

    fig_stable_rank(args.datadir, os.path.join(args.outdir, "lora_stable_rank_spectrum"))
    fig_energy_capture(args.datadir, args.specdir, os.path.join(args.outdir, "lora_energy_capture_bound"))
    fig_energy_capture_exact(args.specdir, os.path.join(args.outdir, "lora_energy_capture_exact"))
    fig_margin_displacement(args.evaldir, os.path.join(args.outdir, "lora_margin_displacement"))


if __name__ == "__main__":
    main()

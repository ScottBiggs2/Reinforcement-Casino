"""
E2: does the Theorem 2 certifiability violation rate V(M) predict trajectory divergence d(M)?

For each sparse arm, against a reference arm (the oracle mask's run):

    d(M) = mean_t | L_M(t) - L_oracle(t) |      over the final --final_steps steps

joined on `step`, never on wall-clock. Also emits the paired trace Delta(t) = L_M(t) - L_oracle(t)
with a block-bootstrap CI over contiguous step blocks, so the autocorrelation in a loss curve is
not mistaken for independent samples.

V comes from mask_score_gap_gap_diagnostics.json as 1 - cert_strict_frac_*, and is only
comparable across arms when the estimator matches how the masks were actually built, i.e. under
cert_tau_rule=hybrid_global_phase with cert_hybrid_min_layer_keep_ratio equal to the training
min_layer_keep_ratio.

Two gates run before any differencing, and both refuse rather than warn:

  1. Config drift. Arms are compared field-by-field from training_args.bin (run_manifest.json
     carries only ~12 fields and lacks warmup_ratio, seed, max_grad_norm, lr_scheduler_type).
     Run names are not trustworthy; drift here is silent and real.
  2. Step-1 grad_norm equality. If two arms disagree at step 1, their data order or init diverged
     and the pair is excluded from d entirely — not reported with a caveat.

Usage:

  python src/analysis/trajectory_divergence.py \
    --reference oracle_deltalog=/scratch/$USER/rl_casino_sparse_train/2026..._oracle_deltalog \
    --arm warm_k250=/scratch/$USER/rl_casino_sparse_train/2026..._magnitude \
    --arm random=/scratch/$USER/rl_casino_sparse_train/20260728_185437_8811878 \
    --v_json /scratch/$USER/.../mask_score_gap_gap_diagnostics.json \
    --v_key warm_k250=cert_strict_frac_magnitude_step250 \
    --v_key random=cert_strict_frac_random_seed42 \
    --out_dir /scratch/$USER/rl_casino_analysis/rebuttal_e1_e2/e2
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

# Compared exactly. `optim` is deliberately absent: sparse arms log adamw_torch_fused and the
# dense arm adamw_8bit, but for sparse arms TrainingArguments.optim is never consulted — the real
# optimizer is SparseAdamW passed via optimizers= (sparse_dpo_bsr.py:282). It is constant within
# the sparse family and would only produce a false alarm.
GUARD_FIELDS = [
    "max_steps",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "max_grad_norm",
    "gradient_accumulation_steps",
    "per_device_train_batch_size",
    "beta",
    "seed",
    "data_seed",
    "max_length",
]


def find_run_dir(path):
    """Resolve a run directory: the one containing training_log.csv, at this level or one below."""
    if os.path.isfile(os.path.join(path, "training_log.csv")):
        return path
    hits = sorted(glob.glob(os.path.join(path, "*", "training_log.csv")))
    if len(hits) == 1:
        return os.path.dirname(hits[0])
    if not hits:
        raise SystemExit(f"no training_log.csv under {path}")
    raise SystemExit(f"ambiguous: {len(hits)} training_log.csv under {path}:\n  " +
                     "\n  ".join(hits))


def read_training_log(run_dir):
    """step -> row dict, from CSVLoggerCallback output.

    Two quirks of that callback are handled here. Its header is fixed by the *first* log dict and
    csv.writer emits .values() positionally, so the trailing train_runtime summary row is
    misaligned against the header (it shows up as a second row at the final step, with the runtime
    in the loss column). Keeping the first row per step drops it. Rows with the wrong field count
    are dropped too.
    """
    path = os.path.join(run_dir, "training_log.csv")
    out, dropped = {}, 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n = len(header)
        for raw in reader:
            if len(raw) != n:
                dropped += 1
                continue
            row = dict(zip(header, raw))
            try:
                step = int(row["step"])
            except (KeyError, ValueError):
                dropped += 1
                continue
            if step in out:  # the summary row; the real log row for this step came first
                dropped += 1
                continue
            rec = {}
            for k, v in row.items():
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
            rec["step"] = step
            out[step] = rec
    return out, dropped


def read_training_args(run_dir):
    """The pickled TrainingArguments/DPOConfig from the newest checkpoint in this run."""
    cands = sorted(
        glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint-*", "training_args.bin")),
        key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]),
    )
    if not cands:
        cands = sorted(glob.glob(os.path.join(run_dir, "**", "training_args.bin"), recursive=True))
    if not cands:
        return None, None
    path = cands[-1]
    # weights_only=False is required: this is a pickled dataclass, not tensors.
    args_obj = torch.load(path, map_location="cpu", weights_only=False)
    fields = {}
    for name in GUARD_FIELDS:
        val = getattr(args_obj, name, "<absent>")
        fields[name] = str(val) if not isinstance(val, (int, float, bool, type(None))) else val
    acc = getattr(args_obj, "accelerator_config", None)
    seedable = getattr(acc, "use_seedable_sampler", None)
    if seedable is None and isinstance(acc, dict):
        seedable = acc.get("use_seedable_sampler")
    fields["accelerator_config.use_seedable_sampler"] = seedable
    fields["optim"] = str(getattr(args_obj, "optim", "<absent>"))  # reported, not guarded
    return fields, path


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
    return a == b


def config_guard(ref_label, ref_fields, label, fields):
    """Field-by-field comparison. Returns the list of guarded mismatches (empty = pass)."""
    bad = []
    if ref_fields is None or fields is None:
        bad.append(("<training_args.bin>", "present" if ref_fields else "missing",
                    "present" if fields else "missing"))
        return bad
    for key in GUARD_FIELDS + ["accelerator_config.use_seedable_sampler"]:
        a, b = ref_fields.get(key, "<absent>"), fields.get(key, "<absent>")
        if not _same(a, b):
            bad.append((key, a, b))
    return bad


def block_bootstrap(values, block=10, n_boot=10000, seed=42):
    """CI for the mean of an autocorrelated series, resampling contiguous blocks."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    idx = np.clip(idx[:, :n], 0, n - 1)
    means = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "block": block,
        "n_boot": n_boot,
        "n": n,
    }


def analyze_pair(ref_log, arm_log, final_steps, block, n_boot):
    """Delta(t) on the shared step grid, and d over the final window."""
    steps = sorted(set(ref_log) & set(arm_log))
    if not steps:
        raise SystemExit("no shared steps between reference and arm")
    delta = [(s, arm_log[s]["loss"] - ref_log[s]["loss"]) for s in steps
             if isinstance(arm_log[s].get("loss"), float) and isinstance(ref_log[s].get("loss"), float)]
    tail = [d for s, d in delta if s > max(steps) - final_steps]
    return {
        "n_shared_steps": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "window": [max(steps) - final_steps + 1, max(steps)],
        "d": float(np.mean(np.abs(tail))) if tail else None,
        "d_ci": block_bootstrap(np.abs(tail), block, n_boot),
        "signed_delta_ci": block_bootstrap(tail, block, n_boot),
        "max_abs_delta_window": float(np.max(np.abs(tail))) if tail else None,
        "trace_steps": [s for s, _ in delta],
        "trace_delta": [d for _, d in delta],
    }


def plot_v_vs_d(results, v_only, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = [r["V"] for r in results if r["V"] is not None and r["d"] is not None]
    ys = [r["d"] for r in results if r["V"] is not None and r["d"] is not None]
    labels = [r["label"] for r in results if r["V"] is not None and r["d"] is not None]
    for r in results:
        if r["V"] is None or r["d"] is None:
            continue
        lo = r["d_ci"]["ci_lo"] if r.get("d_ci") else r["d"]
        hi = r["d_ci"]["ci_hi"] if r.get("d_ci") else r["d"]
        ax.errorbar(r["V"], r["d"], yerr=[[r["d"] - lo], [hi - r["d"]]],
                    fmt="o", ms=8, capsize=4, color="C0", zorder=3)
        ax.annotate(r["label"], (r["V"], r["d"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    # Definitional anchor: at k=T, s_warm == s_oracle, so V=0 and d=0 by construction.
    ax.plot([0.0], [0.0], marker="o", ms=9, mfc="none", mec="0.35", mew=1.6, zorder=3)
    ax.annotate("oracle (definitional\nanchor, not a data point)", (0.0, 0.0),
                textcoords="offset points", xytext=(10, 4), fontsize=8, color="0.35")
    for label, v in v_only.items():
        ax.axvline(v, ls=":", lw=1.0, color="0.6", zorder=1)
        ax.annotate(f"{label}\n(V only)", (v, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(3, -26), fontsize=7.5, color="0.45")
    ax.set_xlabel("V(M) = fraction of weights violating the Theorem 2 condition")
    ax.set_ylabel(f"d(M) = mean |L_M - L_oracle| over the final steps")
    ax.set_title("Does certifiability predict trajectory divergence?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"v_vs_d.{ext}"), dpi=175)
    plt.close(fig)


def plot_v_flatness(v_by_k, v_random, out_dir):
    if not v_by_k:
        return
    ks = sorted(v_by_k)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ks, [v_by_k[k] for k in ks], "o-", label="warm-start (magnitude) mask")
    if v_random is not None:
        ax.axhline(v_random, ls="--", color="C3", label=f"random mask (V={v_random:.4f})")
    ax.set_xlabel("warm-start milestone k (steps)")
    ax.set_ylabel("V(M)")
    ax.set_title("V across k: flat within the warm family, separated from random")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"v_flatness.{ext}"), dpi=175)
    plt.close(fig)


def plot_traces(results, final_steps, out_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i, r in enumerate(results):
        steps = np.asarray(r["trace_steps"])
        d = np.asarray(r["trace_delta"])
        ax.plot(steps, d, lw=0.7, alpha=0.35, color=f"C{i}")
        # Rolling mean makes the level visible through the per-step noise.
        w = 15
        if len(d) >= w:
            k = np.ones(w) / w
            ax.plot(steps[w - 1:], np.convolve(d, k, mode="valid"), lw=1.8, color=f"C{i}",
                    label=f"{r['label']}  (d={r['d']:.4f})")
        else:
            ax.plot(steps, d, lw=1.8, color=f"C{i}", label=r["label"])
        if r.get("signed_delta_ci"):
            ci = r["signed_delta_ci"]
            ax.fill_between(r["window"], ci["ci_lo"], ci["ci_hi"], color=f"C{i}", alpha=0.18,
                            lw=0)
    ax.axhline(0.0, color="0.3", lw=1.0)
    lo = results[0]["window"][0] if results else 0
    ax.axvspan(lo, results[0]["window"][1] if results else 0, color="0.85", alpha=0.35, zorder=0)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\Delta(t) = L_M(t) - L_{oracle}(t)$")
    ax.set_title(f"Paired loss difference vs the oracle-mask run "
                 f"(shaded: final {final_steps} steps, with block-bootstrap CI of the mean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"delta_traces.{ext}"), dpi=175)
    plt.close(fig)


def parse_kv(specs, what):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--{what} expects label=value, got {spec!r}")
        k, v = spec.split("=", 1)
        out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="label=run_dir for the oracle-mask run")
    p.add_argument("--arm", action="append", default=[], help="label=run_dir, repeatable")
    p.add_argument("--v_json", default=None, help="mask_score_gap_gap_diagnostics.json")
    p.add_argument("--v_key", action="append", default=[],
                   help="label=<cert_strict_frac_* key in the diagnostics json>, repeatable")
    p.add_argument("--v_only", action="append", default=[],
                   help="label=<cert_strict_frac_* key> for V-only points (no paired d)")
    p.add_argument("--final_steps", type=int, default=100)
    p.add_argument("--bootstrap_block", type=int, default=10)
    p.add_argument("--bootstrap_draws", type=int, default=10000)
    p.add_argument("--allow_failed_gate", action="store_true",
                   help="Report gate failures without excluding the arm. Off by default on "
                        "purpose: a failed gate means the pairing is invalid, not noisy.")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ref_label, ref_path = parse_kv([args.reference], "reference").popitem()
    ref_dir = find_run_dir(ref_path)
    ref_log, ref_dropped = read_training_log(ref_dir)
    ref_fields, ref_args_path = read_training_args(ref_dir)
    print(f"reference {ref_label}: {ref_dir}")
    print(f"  {len(ref_log)} steps, {ref_dropped} rows dropped, training_args={ref_args_path}")

    v_all = {}
    if args.v_json:
        with open(args.v_json) as f:
            diag = json.load(f)
        for label, key in parse_kv(args.v_key, "v_key").items():
            if key not in diag:
                raise SystemExit(f"key {key!r} absent from {args.v_json}; "
                                 f"available cert_strict_frac keys: "
                                 f"{[k for k in diag if k.startswith('cert_strict_frac')]}")
            v_all[label] = 1.0 - float(diag[key])
        v_only = {}
        for label, key in parse_kv(args.v_only, "v_only").items():
            if key not in diag:
                raise SystemExit(f"key {key!r} absent from {args.v_json}")
            v_only[label] = 1.0 - float(diag[key])
    else:
        diag, v_only = {}, {}

    results, excluded = [], []
    for label, path in parse_kv(args.arm, "arm").items():
        run_dir = find_run_dir(path)
        log, dropped = read_training_log(run_dir)
        fields, args_path = read_training_args(run_dir)
        print(f"\narm {label}: {run_dir}")
        print(f"  {len(log)} steps, {dropped} rows dropped, training_args={args_path}")

        gate_msgs = []
        bad = config_guard(ref_label, ref_fields, label, fields)
        if bad:
            for key, a, b in bad:
                gate_msgs.append(f"config drift {key}: reference={a!r} vs {label}={b!r}")

        s1_ref = ref_log.get(1, {}).get("grad_norm")
        s1_arm = log.get(1, {}).get("grad_norm")
        if s1_ref is None or s1_arm is None:
            gate_msgs.append(f"step-1 grad_norm missing (ref={s1_ref}, {label}={s1_arm})")
        elif not _same(float(s1_ref), float(s1_arm)):
            gate_msgs.append(
                f"step-1 grad_norm mismatch: reference={s1_ref} vs {label}={s1_arm} "
                f"(data order or init diverged; pairing invalid)")

        if fields is not None and ref_fields is not None and fields.get("optim") != ref_fields.get("optim"):
            print(f"  NOTE optim differs ({ref_fields.get('optim')} vs {fields.get('optim')}) — "
                  f"whitelisted; sparse arms never consult TrainingArguments.optim.")

        if gate_msgs:
            for m in gate_msgs:
                print(f"  GATE FAIL: {m}")
            if not args.allow_failed_gate:
                excluded.append({"label": label, "run_dir": run_dir, "reasons": gate_msgs})
                print(f"  -> {label} EXCLUDED from d (not caveated)")
                continue
            print(f"  -> --allow_failed_gate set; keeping {label} with reasons recorded")
        else:
            print(f"  gates PASS (config guard + step-1 grad_norm={s1_arm})")

        res = analyze_pair(ref_log, log, args.final_steps, args.bootstrap_block,
                           args.bootstrap_draws)
        res.update({
            "label": label,
            "run_dir": run_dir,
            "V": v_all.get(label),
            "step1_grad_norm": s1_arm,
            "gate_warnings": gate_msgs,
            "final_loss": log[max(log)]["loss"] if log else None,
        })
        results.append(res)
        ci = res["d_ci"]
        print(f"  d={res['d']:.6f}  95% CI [{ci['ci_lo']:.6f}, {ci['ci_hi']:.6f}] "
              f"(block={ci['block']}, n={ci['n']})   V={res['V']}")

    if results:
        plot_v_vs_d(results, v_only, args.out_dir)
        plot_traces(results, args.final_steps, args.out_dir)

    v_by_k, v_random = {}, None
    for label, v in list(v_all.items()) + list(v_only.items()):
        if "random" in label:
            v_random = v
        else:
            digits = "".join(ch for ch in label if ch.isdigit())
            if digits:
                v_by_k[int(digits)] = v
    plot_v_flatness(v_by_k, v_random, args.out_dir)

    summary = {
        "reference": {"label": ref_label, "run_dir": ref_dir, "training_args": ref_args_path,
                      "fields": ref_fields, "step1_grad_norm": ref_log.get(1, {}).get("grad_norm")},
        "final_steps": args.final_steps,
        "v_json": args.v_json,
        "v_measured": v_all,
        "v_only": v_only,
        "v_by_k": {str(k): v for k, v in sorted(v_by_k.items())},
        "arms": [{k: v for k, v in r.items() if k not in ("trace_steps", "trace_delta")}
                 for r in results],
        "excluded_arms": excluded,
        "note": ("The oracle reference contributes V=0, d=0 by construction (at k=T, "
                 "s_warm == s_oracle) and is a definitional anchor, not a measured point. "
                 "No regression coefficient is reported on this many points."),
    }
    with open(os.path.join(args.out_dir, "e2_divergence.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "e2_delta_traces.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "delta_loss"])
        for r in results:
            for s, d in zip(r["trace_steps"], r["trace_delta"]):
                w.writerow([r["label"], s, f"{d:.10g}"])

    print("\n=== summary ===")
    print(f"{'arm':<16} {'V':>9} {'d':>10} {'d 95% CI':>26} {'final loss':>11}")
    for r in sorted(results, key=lambda r: (r["V"] is None, r["V"])):
        ci = r["d_ci"]
        vs = f"{r['V']:.5f}" if r["V"] is not None else "n/a"
        print(f"{r['label']:<16} {vs:>9} {r['d']:>10.6f} "
              f"[{ci['ci_lo']:.6f}, {ci['ci_hi']:.6f}]".rjust(26) +
              f" {r['final_loss']:>11.4f}")
    if excluded:
        print(f"\nexcluded arms: {[e['label'] for e in excluded]}")
    print(f"\nwrote {args.out_dir}/e2_divergence.json, e2_delta_traces.csv, "
          f"v_vs_d.png, v_flatness.png, delta_traces.png")


if __name__ == "__main__":
    main()

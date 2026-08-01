#!/usr/bin/env python3
r"""Settle whether tau_hat at a given rho is a real boundary or the tie-break floor.

Independent of ``streaming_exact_kth_largest``: this counts, exactly, how many post-floor
coordinates exceed each of a ladder of thresholds. That bounds tau_hat without any histogram
narrowing, so it is a genuine cross-check rather than a restatement.

The decisive comparison, per (arm, rho):

    R = keep - sum(floors)                      the global-phase budget
    count(post-floor value > t)  for t in the ladder

  * if count(> 1e-8) >= R, then tau_hat >= 1e-8 and any reported tau_hat ~1e-15 is WRONG;
  * if count(> 1e-8)  < R, then the R-th largest genuinely lies in the tie-break noise, so
    tau_hat ~1e-15 is CORRECT and the degeneracy flag (raw support < keep) is the wrong test --
    the right one is survivors < R.

Also reports per-tensor floor accounting, since the hybrid floor is per-tensor and the signal is
concentrated in attention while the floors are proportional to tensor size.

Reads only the magnitude cache (or the two checkpoints for the oracle arm) -- no recompute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.certifiability_margin import (  # noqa: E402
    add_tie_break_noise_,
    global_keep_count,
    scaled_hybrid_floor_counts_per_layer,
    tie_break_scale,
)
from src.analysis.certifiability_margins import (  # noqa: E402
    guard_not_under_home,
    resolve_scope,
    safe_torch_load,
    stream_seed,
)
from src.warm_start.checkpoint_diff_mask_finder import load_state_dict  # noqa: E402

LADDER = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-13, 1e-14]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial_model", required=True)
    ap.add_argument("--final_model", required=True)
    ap.add_argument("--cache", default=None, help="mag_aggregate_step_<k>.pt; omit for the oracle arm")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sparsity_percents", default="97.5,99,99.5")
    ap.add_argument("--hybrid_min_layer_keep_ratio", type=float, default=0.0025)
    ap.add_argument("--tie_break_relative_scale", type=float, default=1e-12)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    guard_not_under_home(Path(args.out_json).parent, label="--out_json dir", allow=False)
    rhos = [float(x) for x in args.sparsity_percents.split(",") if x.strip()]

    initial_sd = load_state_dict(args.initial_model, device="cpu", torch_dtype=torch.bfloat16)
    final_sd = load_state_dict(args.final_model, device="cpu", torch_dtype=torch.bfloat16)
    names = resolve_scope(
        initial_sd, final_sd, score_scope="builder_2d", scope_source="initial", mlp_only=False, max_keys=None
    )
    numels = [int(initial_sd[n].numel()) for n in names]
    n_total = int(sum(numels))
    print(f"scope: {len(names)} tensors, {n_total} elements", flush=True)

    mag = safe_torch_load(args.cache) if args.cache else None

    def raw(name: str) -> torch.Tensor:
        if mag is not None:
            return mag[name].to(torch.float64).reshape(-1)
        return (final_sd[name].to(torch.float64) - initial_sd[name].to(torch.float64)).abs().reshape(-1)

    # pass 1: max|s| (sets the tie-break amplitude) and raw support
    max_abs = 0.0
    n_pos_total = 0
    for n in names:
        x = raw(n)
        max_abs = max(max_abs, float(x.abs().max().item()))
        n_pos_total += int((x > 0).sum().item())
        del x
    scale = tie_break_scale(max_abs, args.tie_break_relative_scale)
    print(f"max|s|={max_abs:.6e}  raw_support={n_pos_total}  tie_break_scale={scale:.6e}", flush=True)

    out: Dict[str, object] = {
        "arm": args.arm,
        "cache": args.cache,
        "n_tensors": len(names),
        "n_total": n_total,
        "max_abs": max_abs,
        "raw_support": n_pos_total,
        "tie_break_scale": scale,
        "per_rho": {},
    }

    for rho in rhos:
        keep = global_keep_count(n_total, rho)
        floors = scaled_hybrid_floor_counts_per_layer(numels, args.hybrid_min_layer_keep_ratio, keep)
        floor_by = {n: int(f) for n, f in zip(names, floors)}
        floor_total = int(sum(floors))
        R = keep - floor_total

        ladder_asc = torch.tensor(sorted(LADDER), dtype=torch.float64)
        above = {t: 0 for t in ladder_asc.tolist()}
        survivors = 0          # positives left after the floors
        pos_eaten = 0          # positives consumed by the floors
        floor_slots_on_zeros = 0
        worst: List[Dict[str, object]] = []

        for n in names:
            x = raw(n)
            n_pos = int((x > 0).sum().item())
            f = floor_by.get(n, 0)
            # The floor takes the top-f of the *noised* score, so it eats min(f, n_pos) positives
            # and spends the rest of its slots on zero-score coordinates.
            eaten = min(f, n_pos)
            pos_eaten += eaten
            floor_slots_on_zeros += max(0, f - n_pos)
            survivors += n_pos - eaten

            sel = x.clone()
            if scale > 0:
                g = torch.Generator(device="cpu")
                g.manual_seed(stream_seed("tie_break", "warm" if mag is not None else "oracle", 0, n))
                add_tie_break_noise_(sel, scale=scale, generator=g)
            if f > 0:
                _, idx = torch.topk(sel, min(f, int(sel.numel())), largest=True)
                sel[idx] = float("-inf")
            # All ladder counts in one pass: bucketize gives, per value, how many ladder entries it
            # exceeds; the suffix sums of that histogram are count(> t) for every t at once.
            b = torch.bucketize(sel, ladder_asc, right=True)
            hist = torch.bincount(b, minlength=len(LADDER) + 1)
            suffix = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
            for j, t in enumerate(ladder_asc.tolist()):
                above[t] += int(suffix[j + 1].item())
            if len(worst) < 6:
                worst.append({"tensor": n, "numel": int(x.numel()), "n_pos": n_pos, "floor": f, "eaten": eaten})
            del x, sel

        # tau_hat is bracketed by the ladder: the largest t with count(>t) >= R is a lower bound.
        # Largest ladder threshold whose count still reaches R is a lower bound on tau_hat.
        lower = None
        for t in sorted(above, reverse=True):
            if above[t] >= R:
                lower = t
                break
        verdict = (
            f"tau_hat >= {lower:g} (count(> {lower:g}) = {above[lower]} >= R = {R})"
            if lower is not None
            else f"tau_hat sits in the tie-break noise: even count(> {LADDER[-1]:g}) = {above[LADDER[-1]]} < R = {R}"
        )
        rec = {
            "keep": keep,
            "floor_total": floor_total,
            "R": R,
            "positives_eaten_by_floors": pos_eaten,
            "floor_slots_spent_on_zeros": floor_slots_on_zeros,
            "survivors": survivors,
            "survivors_ge_R": bool(survivors >= R),
            "raw_support_ge_keep": bool(n_pos_total >= keep),
            "count_above": {f"{t:g}": above[t] for t in sorted(above, reverse=True)},
            "tau_lower_bound": lower,
            "verdict": verdict,
        }
        out["per_rho"][f"{rho:g}"] = rec  # type: ignore[index]
        print(
            f"\nrho={rho:g}: keep={keep} floors={floor_total} R={R}\n"
            f"  raw_support={n_pos_total} ({'>=' if n_pos_total >= keep else '<'} keep)  "
            f"survivors={survivors} ({'>=' if survivors >= R else '<'} R)\n"
            f"  floors ate {pos_eaten} positives, spent {floor_slots_on_zeros} slots on zeros\n"
            f"  {verdict}",
            flush=True,
        )
        print("  count(> t): " + "  ".join(f"{t:g}:{above[t]}" for t in sorted(above, reverse=True)), flush=True)

    out["sample_tensors"] = worst
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()

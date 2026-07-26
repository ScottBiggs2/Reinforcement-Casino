"""Two properties of the weight update dtheta = theta_T - theta_0, from checkpoints.

(A) MASS CONCENTRATION -- what fraction of ||dtheta||^2 lives in the top-k% of
    coordinates. This is what makes the "GRPO tolerates less sparsity than DPO"
    claim a measurement instead of a rationalisation: run it on the dense DPO and
    dense GRPO checkpoints, and the viable sparsity threshold becomes a prediction
    read off a measured curve, with both objectives on the same axis.

(B) STABLE RANK of each dW -- ||dW||_F^2 / ||dW||_2^2, a smooth surrogate for rank.
    LoRA at rank r caps this at r by construction. If the 97.5%-sparse dW has stable
    rank >> 64 at a matched trainable-parameter budget, that is the mechanistic answer
    to "why not just use LoRA" -- and unlike an accuracy table, it survives the two
    methods tying on quality. The paper's premise (Mukherjee et al.) is that RL updates
    are sparse but full-rank; this measures it directly on our own checkpoints.

Loads two checkpoints one tensor at a time, so peak memory is one tensor pair rather
than two 8B state dicts.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch


def iter_named_tensors(path: str):
    """Yield (name, tensor) from an HF checkpoint dir or a .pt state dict, lazily."""
    import os
    # A local .pt state dict is the only case that goes through torch.load. Anything
    # else — a local HF checkpoint directory OR a hub id like "meta-llama/Llama-3.1-8B-
    # Instruct" — goes through from_pretrained. Testing isdir() alone sent hub ids into
    # torch.load and failed with FileNotFoundError.
    is_local_file = os.path.isfile(path)
    if not is_local_file:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu",
        )
        for n, p in model.named_parameters():
            yield n, p.detach()
        del model
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        for n, t in sd.items():
            yield n, t


def concentration_curve(sq: torch.Tensor, fractions: List[float]) -> Dict[str, float]:
    """Fraction of total squared mass held by the top-f coordinates, for each f."""
    total = sq.sum().item()
    if total <= 0:
        return {f"top_{f*100:g}pct": float("nan") for f in fractions}
    n = sq.numel()
    out = {}
    # one sort, then prefix sums -- cheaper than a topk per fraction
    vals, _ = torch.sort(sq, descending=True)
    csum = torch.cumsum(vals.double(), dim=0)
    for f in fractions:
        k = max(1, int(round(f * n)))
        out[f"top_{f*100:g}pct"] = (csum[k - 1] / total).item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="dtheta mass concentration + stable rank")
    ap.add_argument("--initial", required=True, help="theta_0: base model dir or state dict")
    ap.add_argument("--final", required=True, help="theta_T: trained checkpoint dir or state dict")
    ap.add_argument("--fractions", default="0.0025,0.01,0.025,0.05,0.10,0.25",
                    help="Comma-separated top-k fractions for the concentration curve.")
    ap.add_argument("--stable_rank", action="store_true",
                    help="Also compute per-matrix stable rank of dW (2-D tensors only).")
    ap.add_argument("--rank_max_tensors", type=int, default=64,
                    help="Cap on how many 2-D tensors get a stable rank (SVD-free but O(n^2) matmul).")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]

    print(f"Loading final:   {args.final}")
    final = dict(iter_named_tensors(args.final))
    print(f"Loading initial: {args.initial}")

    per_layer: List[Dict[str, object]] = []
    ranks: List[Dict[str, object]] = []
    global_sq_chunks: List[torch.Tensor] = []
    total_sq = 0.0
    n_params = 0
    n_ranked = 0

    for name, t0 in iter_named_tensors(args.initial):
        if name not in final:
            continue
        d = (final[name].float() - t0.float()).flatten()
        sq = d * d
        s = sq.sum().item()
        total_sq += s
        n_params += d.numel()

        per_layer.append({
            "name": name,
            "numel": d.numel(),
            "sq_mass": s,
            **concentration_curve(sq, fractions),
        })

        # keep squared magnitudes for the global curve; float32 over 8B coords is ~32GB,
        # so subsample uniformly above a threshold rather than materialising everything
        if sq.numel() > 2_000_000:
            idx = torch.randperm(sq.numel())[:2_000_000]
            global_sq_chunks.append(sq[idx] * (sq.numel() / 2_000_000))
        else:
            global_sq_chunks.append(sq)

        if args.stable_rank and final[name].dim() == 2 and n_ranked < args.rank_max_tensors:
            dW = (final[name].float() - t0.float())
            fro2 = (dW * dW).sum().item()
            # spectral norm via power iteration -- full SVD on 4096x14336 is wasteful
            v = torch.randn(dW.shape[1])
            v /= v.norm()
            for _ in range(30):
                u = dW @ v
                u /= (u.norm() + 1e-12)
                v = dW.T @ u
                nv = v.norm()
                v = v / (nv + 1e-12)
            spec2 = (nv ** 2).item()
            ranks.append({
                "name": name,
                "shape": list(dW.shape),
                "stable_rank": fro2 / spec2 if spec2 > 0 else float("nan"),
                "max_possible_rank": min(dW.shape),
            })
            n_ranked += 1
            del dW

        del d, sq

    global_sq = torch.cat(global_sq_chunks)
    del global_sq_chunks
    global_curve = concentration_curve(global_sq, fractions)

    result = {
        "tag": args.tag or args.final,
        "initial": args.initial,
        "final": args.final,
        "n_params_compared": n_params,
        "total_sq_mass": total_sq,
        "global_concentration": global_curve,
        "per_layer": per_layer,
        "stable_ranks": ranks,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n{'=' * 70}")
    print(f"dtheta MASS CONCENTRATION — {result['tag']}")
    print(f"  {n_params:,} parameters compared")
    print(f"{'=' * 70}")
    for k, v in global_curve.items():
        print(f"  {k:>14}  holds {v*100:6.2f}% of ||dtheta||^2")
    if ranks:
        sr = [r["stable_rank"] for r in ranks if r["stable_rank"] == r["stable_rank"]]
        if sr:
            sr_sorted = sorted(sr)
            print(f"\n  stable rank of dW over {len(sr)} matrices:")
            print(f"    min {min(sr):.1f}   median {sr_sorted[len(sr)//2]:.1f}   max {max(sr):.1f}")
            print(f"    (LoRA rank r caps this at r — compare against the r used in the baseline)")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

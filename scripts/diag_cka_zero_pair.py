"""Diagnose the exactly-0.0 CKA between the two GRPO oracle masks (job 240279).

Both masks score normally against every other mask, so neither activation set is
degenerate on its own — the zero is specific to the pair. This dumps the HSIC
components the current estimator computes, plus the activation magnitudes that
feed them, and recomputes CKA in the numerically stable Kornblith form for
comparison.

The suspect: linear_cka builds Kc = H (X Xᵀ) H explicitly. Llama MLP
intermediates carry massive outlier channels, so X Xᵀ entries are enormous while
the centered matrix is tiny — a cancellation of many orders of magnitude. The
stable form never materializes the Gram matrix: it centers the columns of X and
Y and evaluates ‖Yᵀ X‖²_F / (‖Xᵀ X‖_F ‖Yᵀ Y‖_F), which is algebraically identical
and loses no digits.

Usage: python scripts/diag_cka_zero_pair.py MASK_A MASK_B
"""
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from cold_start.mask_to_cka import (  # noqa: E402
    apply_mask,
    collect_activations,
    linear_cka,
    load_calibration_samples,
    load_masks,
    restore_weights,
)
from cold_start.utils.activation_hooks import FeatureExtractor  # noqa: E402


def cka_stable(X, Y):
    """Linear CKA without forming the n×n Gram matrices (Kornblith et al. 2019)."""
    X = X.detach().cpu().double()
    Y = Y.detach().cpu().double()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    num = (Y.t() @ X).norm(p="fro") ** 2
    den = (X.t() @ X).norm(p="fro") * (Y.t() @ Y).norm(p="fro")
    if den.item() == 0.0:
        return float("nan")
    return (num / den).item()


def hsic_parts(X, Y):
    """The exact quantities the current linear_cka computes, before the clamp."""
    X = X.detach().cpu().double()
    Y = Y.detach().cpu().double()
    n = X.shape[0]
    K, L = X @ X.t(), Y @ Y.t()
    H = torch.eye(n, dtype=X.dtype) - torch.ones(n, n, dtype=X.dtype) / n
    Kc, Lc = H @ K @ H, H @ L @ H
    f = lambda A, B: (A * B).sum() / (n - 1) ** 2
    kl, kk, ll = f(Kc, Lc), f(Kc, Kc), f(Lc, Lc)
    return dict(hsic_kl=kl.item(), hsic_kk=kk.item(), hsic_ll=ll.item(),
                denom=(kk * ll).sqrt().item(), K_absmax=K.abs().max().item(),
                Kc_absmax=Kc.abs().max().item())


def main():
    mask_a, mask_b = sys.argv[1], sys.argv[2]
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    chosen, _ = load_calibration_samples(n_samples=64, seed=42)
    texts = chosen[:64]

    ex = FeatureExtractor().register(model)
    acts = {}
    for tag, path in (("A", mask_a), ("B", mask_b)):
        m, _ = load_masks(path)
        apply_mask(model, m)
        t0 = time.time()
        acts[tag] = collect_activations(model, ex, tok, texts, device,
                                        batch_size=4, max_length=512)
        print(f"{tag}: {Path(path).name}  collected {len(acts[tag])} layers "
              f"in {time.time() - t0:.1f}s", flush=True)
        restore_weights(model, original)

    A, B = acts["A"], acts["B"]
    print(f"\nsame object? {A is B}")
    layers = sorted(set(A) & set(B))[:4]
    for name in layers:
        X, Y = A[name], B[name]
        ident = torch.equal(X, Y)
        print(f"\n--- {name}")
        print(f"  X {tuple(X.shape)} absmax={X.abs().max():.4g}  "
              f"Y absmax={Y.abs().max():.4g}  bitwise-identical={ident}")
        print(f"  finite: X={torch.isfinite(X).all().item()} Y={torch.isfinite(Y).all().item()}")
        p = hsic_parts(X, Y)
        for k, v in p.items():
            print(f"  {k:>10} = {v:.6e}")
        print(f"  current linear_cka = {linear_cka(X, Y):.6f}")
        print(f"  stable      CKA    = {cka_stable(X, Y):.6f}")
        print(f"  stable self-CKA(X,X) = {cka_stable(X, X):.6f}  (must be 1.0)")


if __name__ == "__main__":
    main()

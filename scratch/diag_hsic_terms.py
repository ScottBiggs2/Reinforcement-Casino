"""Diagnostic: dump raw HSIC terms per layer to explain CKA=0.0 cases.
Runs from the irene worktree so it uses her mask_to_cka helpers + FeatureExtractor.
"""
import re
import sys
from pathlib import Path

_ROOT = Path("/scratch/biggs.s/irene_wt")
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cold_start.utils.activation_hooks import FeatureExtractor
from cold_start.mask_to_cka import (
    apply_mask, collect_activations, load_calibration_samples,
    load_masks, restore_weights, set_seed,
)

M = "/scratch/biggs.s/rl_casino_grpo/masks/evolstudy"
MASKS = [("mag250", f"{M}/mag_step250.pt"),
         ("oracle", f"{M}/oracle_gt.pt"),
         ("random", f"{M}/random.pt")]

set_seed(42)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
dev = torch.device("cuda")
model.to(dev).eval()
orig = {n: p.detach().clone() for n, p in model.named_parameters()}
chosen, _ = load_calibration_samples(n_samples=32, seed=42, dataset_name="tulu3")
texts = chosen[:32]
ext = FeatureExtractor(); ext.register(model)

acts = {}
for label, path in MASKS:
    mt, _ = load_masks(path)
    apply_mask(model, mt)
    acts[label] = collect_activations(model, ext, tok, texts, dev, batch_size=2, max_length=384)
    restore_weights(model, orig)
    print(f"collected {label}", flush=True)


def hsic_terms(X, Y):
    X = X.detach().cpu().double(); Y = Y.detach().cpu().double()
    n = X.shape[0]
    K = X @ X.t(); L = Y @ Y.t()
    H = torch.eye(n, dtype=X.dtype) - torch.ones(n, n, dtype=X.dtype) / n
    Kc = H @ K @ H; Lc = H @ L @ H
    kl = (Kc * Lc).sum() / (n - 1) ** 2
    kk = (Kc * Kc).sum() / (n - 1) ** 2
    ll = (Lc * Lc).sum() / (n - 1) ** 2
    denom = (kk * ll).sqrt()
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    return dict(kk=kk.item(), ll=ll.item(), kl=kl.item(), denom=denom.item(),
                raw=(kl / denom).item() if denom.abs().item() > 1e-30 else float("nan"),
                xc=Xc.norm().item(), yc=Yc.norm().item(),
                xabs=X.abs().mean().item(), yabs=Y.abs().mean().item())


def lidx(name):
    m = re.search(r"layers?[._](\d+)", name)
    return int(m.group(1)) if m else -1


common = sorted(set(acts["mag250"]) & set(acts["oracle"]), key=lidx)
print("\nlayer | mag250-vs-oracle: kk        ll        kl        denom     raw_ratio | xc(mag)   yc(oracle) | mag-vs-random raw")
for name in common:
    t = hsic_terms(acts["mag250"][name], acts["oracle"][name])
    tr = hsic_terms(acts["mag250"][name], acts["random"][name])
    print(f"{lidx(name):5d} | {t['kk']:.3e} {t['ll']:.3e} {t['kl']:+.3e} {t['denom']:.3e} {t['raw']:+.4f} | "
          f"{t['xc']:.3e} {t['yc']:.3e} | {tr['raw']:+.4f}")

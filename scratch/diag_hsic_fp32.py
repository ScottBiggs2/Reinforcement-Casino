"""Diagnostic v2: is the near-zero across-sample variance a bf16 artifact?
Recollect activations with the model in FLOAT32 and report relative centered norm
+ raw (unguarded) CKA ratio per layer, for mag250 vs oracle and vs random.
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
DTYPE = torch.float32   # <-- the change under test
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=DTYPE, low_cpu_mem_usage=True)
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
    print(f"collected {label} (dtype={DTYPE})", flush=True)


def stats(X, Y):
    X = X.detach().cpu().double(); Y = Y.detach().cpu().double()
    n = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    rel_x = (Xc.norm() / (X.norm() + 1e-30)).item()   # across-sample variation / total magnitude
    K = X @ X.t(); L = Y @ Y.t()
    H = torch.eye(n, dtype=X.dtype) - torch.ones(n, n, dtype=X.dtype) / n
    Kc = H @ K @ H; Lc = H @ L @ H
    kl = (Kc * Lc).sum(); kk = (Kc * Kc).sum(); ll = (Lc * Lc).sum()
    denom = (kk * ll).sqrt()
    raw = (kl / denom).clamp(0, 1).item() if denom.item() > 0 else float("nan")
    return rel_x, raw


def lidx(name):
    m = re.search(r"layers?[._](\d+)", name)
    return int(m.group(1)) if m else -1


common = sorted(set(acts["mag250"]) & set(acts["oracle"]), key=lidx)
print("\nlayer | rel_var(mag) | CKA(mag,oracle) UNGUARDED | CKA(mag,random) UNGUARDED")
for name in common:
    relm, cko = stats(acts["mag250"][name], acts["oracle"][name])
    _, ckr = stats(acts["mag250"][name], acts["random"][name])
    print(f"{lidx(name):5d} | {relm:.3e}     | {cko:.4f}                     | {ckr:.4f}")

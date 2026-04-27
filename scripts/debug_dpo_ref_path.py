#!/usr/bin/env python3
"""
Reproduce TRL DPOTrainer's reference-model path with sparse-wrapped model.

The pure-step debug (debug_sparse_step.py) showed forward+backward+optimizer
all clean. The multi-step debug (debug_sparse_multistep.py) showed 5 optim
steps clean with LM loss. But TRL DPOTrainer convergence (job 6338691)
went to NaN immediately. The remaining suspect is the reference model: TRL
deepcopies the policy and runs both `policy.forward()` and `ref.forward()`
to compute the DPO log-ratio.

This script:
  1. Builds the sparse-wrapped policy model.
  2. Creates a reference via copy.deepcopy + freezes all params.
  3. Runs forward on the SAME input through both, compares logits.
  4. Computes a DPO-style log-ratio loss and checks NaN at each substep.

If policy and reference logits diverge → SparseLinearFunction misbehaves
under `requires_grad=False` (or under deepcopy). If both clean and DPO
loss is NaN → log-ratio computation has a problem with bf16 / sparse weights.
"""

import os
import sys
import copy
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.mlps.bsr_sparse_mlp import replace_linear_modules
from src.utils.mask_manager import SparseMaskManager


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_CHOSEN = "Q: What is 2+2? A: 4"
PROMPT_REJECTED = "Q: What is 2+2? A: 5"
SEED = 42


def has_nan(t):
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()


def stat(label, t):
    if has_nan(t):
        nan = torch.isnan(t).sum().item()
        inf = torch.isinf(t).sum().item()
        print(f"  ❌ {label}: NaN={nan} Inf={inf}")
        return True
    f = t.float()
    print(f"  ✓ {label}: mean={f.mean().item():+.4e} max={f.max().item():+.4e}")
    return False


def build_block_mask(shape, sparsity, block_size, device):
    out_dim, in_dim = int(shape[0]), int(shape[1])
    g = torch.Generator(device="cpu").manual_seed(SEED)
    nb_out = max(1, out_dim // block_size)
    nb_in = max(1, in_dim // block_size)
    n = nb_out * nb_in
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.randperm(n, generator=g)[:max(1, int(round((1 - sparsity) * n)))]] = True
    keep = keep.view(nb_out, nb_in)
    m = keep.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return m[:out_dim, :in_dim].contiguous().to(device)


def main():
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    )
    model.to(device)
    model.config.use_cache = False
    model.train()
    print(f"  on {device} dtype={model.dtype}")

    masks = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if "mlp" not in name.lower():
            continue
        masks[name] = build_block_mask(p.shape, 0.9, 16, p.device)
    print(f"  built {len(masks)} masks")

    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(masks, tmp)
    try:
        mm = SparseMaskManager(tmp, device=device)
        replace_linear_modules(model, mm.masks, block_size=16, use_tf32=True, verbose=False)

        # ==== Build reference like TRL: deepcopy + freeze ====
        print("\n=== Building reference (deepcopy + freeze, mimicking TRL) ===")
        ref_model = copy.deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        print("  done")

        # ==== Tokenize chosen + rejected ====
        enc_chosen = tokenizer(PROMPT_CHOSEN, return_tensors="pt").to(device)
        enc_rejected = tokenizer(PROMPT_REJECTED, return_tensors="pt").to(device)

        # ==== Run forward on policy ====
        print("\n=== Policy forward ===")
        out_policy_chosen = model(**enc_chosen)
        out_policy_rejected = model(**enc_rejected)
        if stat("policy.logits[chosen]", out_policy_chosen.logits): return
        if stat("policy.logits[rejected]", out_policy_rejected.logits): return

        # ==== Run forward on reference (no_grad) ====
        print("\n=== Reference forward (under torch.no_grad) ===")
        with torch.no_grad():
            out_ref_chosen = ref_model(**enc_chosen)
            out_ref_rejected = ref_model(**enc_rejected)
        if stat("ref.logits[chosen]", out_ref_chosen.logits): return
        if stat("ref.logits[rejected]", out_ref_rejected.logits): return

        # ==== Compare policy vs reference (should be identical at step 0) ====
        print("\n=== Policy vs Reference logits diff (should be ~0 at init) ===")
        d_chosen = (out_policy_chosen.logits.float() - out_ref_chosen.logits.float()).abs()
        d_rejected = (out_policy_rejected.logits.float() - out_ref_rejected.logits.float()).abs()
        print(f"  chosen   absdiff: max={d_chosen.max().item():.4e}  mean={d_chosen.mean().item():.4e}")
        print(f"  rejected absdiff: max={d_rejected.max().item():.4e}  mean={d_rejected.mean().item():.4e}")
        if d_chosen.max().item() > 1e-3 or d_rejected.max().item() > 1e-3:
            print("  ⚠ Policy and reference diverge at step 0 — deepcopy is broken.")

        # ==== Compute DPO-style log-ratio ====
        print("\n=== DPO log-ratio ===")
        # Per-token logp by gathering target token's logit, then summing
        def per_token_logp(logits, input_ids):
            # logits: (B, T, V), input_ids: (B, T)
            logp = F.log_softmax(logits.float(), dim=-1)
            # Gather logp at the actual token id at each position
            ids = input_ids.unsqueeze(-1)
            logp_tokens = logp.gather(-1, ids).squeeze(-1)
            return logp_tokens.sum(dim=-1)  # (B,)

        policy_logp_chosen = per_token_logp(out_policy_chosen.logits, enc_chosen["input_ids"])
        policy_logp_rejected = per_token_logp(out_policy_rejected.logits, enc_rejected["input_ids"])
        ref_logp_chosen = per_token_logp(out_ref_chosen.logits, enc_chosen["input_ids"])
        ref_logp_rejected = per_token_logp(out_ref_rejected.logits, enc_rejected["input_ids"])

        if stat("policy_logp[chosen]", policy_logp_chosen): return
        if stat("policy_logp[rejected]", policy_logp_rejected): return
        if stat("ref_logp[chosen]", ref_logp_chosen): return
        if stat("ref_logp[rejected]", ref_logp_rejected): return

        log_ratio_chosen = policy_logp_chosen - ref_logp_chosen
        log_ratio_rejected = policy_logp_rejected - ref_logp_rejected
        if stat("log_ratio[chosen]", log_ratio_chosen): return
        if stat("log_ratio[rejected]", log_ratio_rejected): return

        beta = 0.1
        logits = beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()
        if stat("dpo_loss", loss.unsqueeze(0)): return

        # ==== Backward ====
        print("\n=== Backward through DPO loss ===")
        loss.backward()
        bad = []
        for name, p in model.named_parameters():
            if p.grad is not None and has_nan(p.grad):
                bad.append(name)
        if bad:
            print(f"  ❌ {len(bad)} grads NaN/Inf, e.g. {bad[:3]}")
            return
        print(f"  ✓ all grads finite")

        print("\n✓ ✓ ✓  Full DPO path with deepcopy reference clean.")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()

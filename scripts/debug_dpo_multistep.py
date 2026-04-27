#!/usr/bin/env python3
"""
Multi-step DPO debug — closest yet to what TRL DPOTrainer does.

Single-step DPO + ref model came out clean (debug_dpo_ref_path). But TRL's
real loop produces NaN at step 1. Remaining differences:
  1. Multiple optim steps (DPO drift over time)
  2. Padded batches with attention_mask (variable-length inputs)
  3. Logp computed only over completion tokens (prompt masked out)

This script does N optim steps of true DPO with:
  - deepcopy frozen reference
  - padded batched inputs
  - completion-only logp (mask prompt tokens)
  - gradient_checkpointing on (matches convergence test)
  - SparseAdamW + block kernel
  - NaN check at every micro substep
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
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
N_STEPS = 5

# A few prompt + chosen + rejected triples (variable length on purpose).
DPO_BATCH = [
    ("Q: 2+2? A:", " 4", " 5"),
    ("Q: capital of France?", " Paris.", " London."),
    ("Q: who wrote Hamlet?", " Shakespeare.", " Dickens."),
    ("Q: largest planet?", " Jupiter, by mass and volume in our solar system.", " Mars."),
]


def has_nan(t):
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()


def first_bad_param(model):
    for name, p in model.named_parameters():
        if has_nan(p):
            return name, "param"
        if p.grad is not None and has_nan(p.grad):
            return name, "grad"
    return None, None


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


def build_dpo_batch(tokenizer, device):
    """Tokenize, pad, return chosen+rejected sequences with completion-only labels."""
    chosen_seqs, chosen_completion_starts = [], []
    rejected_seqs, rejected_completion_starts = [], []
    for prompt, chosen, rejected in DPO_BATCH:
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        c_ids = tokenizer(chosen, add_special_tokens=False).input_ids
        r_ids = tokenizer(rejected, add_special_tokens=False).input_ids
        chosen_seqs.append(p_ids + c_ids)
        chosen_completion_starts.append(len(p_ids))
        rejected_seqs.append(p_ids + r_ids)
        rejected_completion_starts.append(len(p_ids))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def pad(seqs):
        maxlen = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        return ids.to(device), attn.to(device)

    c_ids, c_attn = pad(chosen_seqs)
    r_ids, r_attn = pad(rejected_seqs)

    # Completion mask: 1 for completion tokens, 0 for prompt + padding.
    def comp_mask(seqs, completion_starts):
        maxlen = max(len(s) for s in seqs)
        m = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, (s, start) in enumerate(zip(seqs, completion_starts)):
            m[i, start : len(s)] = 1
        return m.to(device)

    c_comp = comp_mask(chosen_seqs, chosen_completion_starts)
    r_comp = comp_mask(rejected_seqs, rejected_completion_starts)

    return c_ids, c_attn, c_comp, r_ids, r_attn, r_comp


def per_completion_logp(logits, input_ids, completion_mask):
    """Sum log P(token_t | < t) over completion tokens only (TRL-style)."""
    # Shift: logits[t] predicts input_ids[t+1]
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    shifted_comp_mask = completion_mask[:, 1:]

    logp = F.log_softmax(shifted_logits.float(), dim=-1)
    selected = logp.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
    return (selected * shifted_comp_mask).sum(dim=-1)


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
    model.gradient_checkpointing_enable()
    model.train()
    print(f"  on {device} dtype={model.dtype}  grad_ckpt=on")

    masks = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if "mlp" not in name.lower():
            continue
        masks[name] = build_block_mask(p.shape, 0.9, 16, p.device)

    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(masks, tmp)
    try:
        mm = SparseMaskManager(tmp, device=device)
        replace_linear_modules(model, mm.masks, block_size=16, use_tf32=True, verbose=False)

        ref_model = copy.deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        # Reference shouldn't grad-checkpoint (eval mode + no_grad).

        opt = SparseAdamW(
            model.named_parameters(),
            mask_manager=mm,
            lr=5e-7,
            block_size=16,
            mlp_only=True,
        )

        c_ids, c_attn, c_comp, r_ids, r_attn, r_comp = build_dpo_batch(tokenizer, device)
        print(f"  batch shapes: c={tuple(c_ids.shape)} r={tuple(r_ids.shape)}")

        beta = 0.1
        for step in range(1, N_STEPS + 1):
            print(f"\n========== DPO STEP {step} ==========")
            opt.zero_grad()

            # Policy forward (chosen + rejected together)
            out_p_c = model(input_ids=c_ids, attention_mask=c_attn)
            out_p_r = model(input_ids=r_ids, attention_mask=r_attn)
            if has_nan(out_p_c.logits) or has_nan(out_p_r.logits):
                print(f"  ❌ policy logits NaN at step {step}")
                return

            # Reference forward (no_grad)
            with torch.no_grad():
                out_r_c = ref_model(input_ids=c_ids, attention_mask=c_attn)
                out_r_r = ref_model(input_ids=r_ids, attention_mask=r_attn)
            if has_nan(out_r_c.logits) or has_nan(out_r_r.logits):
                print(f"  ❌ ref logits NaN at step {step}")
                return

            # Logps over completion only
            p_logp_c = per_completion_logp(out_p_c.logits, c_ids, c_comp)
            p_logp_r = per_completion_logp(out_p_r.logits, r_ids, r_comp)
            r_logp_c = per_completion_logp(out_r_c.logits, c_ids, c_comp).detach()
            r_logp_r = per_completion_logp(out_r_r.logits, r_ids, r_comp).detach()

            log_ratio_c = p_logp_c - r_logp_c
            log_ratio_r = p_logp_r - r_logp_r

            logits = beta * (log_ratio_c - log_ratio_r)
            loss = -F.logsigmoid(logits).mean()
            print(f"  loss={loss.item():.4e}  log_ratio_c={log_ratio_c.mean().item():+.4e}  log_ratio_r={log_ratio_r.mean().item():+.4e}")

            if has_nan(loss):
                print(f"  ❌ loss NaN at step {step}")
                return

            loss.backward()
            bad, kind = first_bad_param(model)
            if bad:
                print(f"  ❌ {kind} NaN/Inf in {bad}")
                return

            opt.step()
            bad, kind = first_bad_param(model)
            if bad:
                print(f"  ❌ after optimizer: {kind} NaN/Inf in {bad}")
                return

        print(f"\n✓ ✓ ✓  {N_STEPS} steps of DPO with sparse all clean.")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()

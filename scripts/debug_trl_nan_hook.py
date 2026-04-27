#!/usr/bin/env python3
"""
Run TRL DPOTrainer with NaN-detection hooks attached to every module.

Multi-step manual DPO came out clean (debug_dpo_multistep.py), so the bug
is somewhere TRL-specific: collator, accelerate wrapping, autocast,
internal logp computation, etc. This wires forward-pre and forward-post
hooks on every submodule, prints a stack trace the FIRST time a module
produces NaN that wasn't in its input, and exits.

That tells us: (a) which module in which sub-tree (policy vs ref?),
(b) whether the NaN is generated locally or just propagating in.
"""

import os
import sys
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from src.mlps.bsr_sparse_mlp import replace_linear_modules
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "qihoo360/Light-R1-DPOData"
SEED = 42

# Module-tree-wide state for the hook to record first NaN.
NAN_FOUND = {"first": None}


def _has_nan(t):
    if not torch.is_tensor(t):
        return False
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()


def _any_nan_in(args):
    """Recursively check if any tensor in (possibly nested) args has NaN."""
    if torch.is_tensor(args):
        return _has_nan(args)
    if isinstance(args, (list, tuple)):
        return any(_any_nan_in(a) for a in args)
    if isinstance(args, dict):
        return any(_any_nan_in(v) for v in args.values())
    return False


def attach_nan_hooks(model: nn.Module, label: str):
    """Hook every module: if output has NaN but inputs didn't, log it once and exit."""
    for name, mod in model.named_modules():
        full = f"{label}.{name}" if name else label

        def make_hook(mod_name):
            def hook(module, args, kwargs, output):
                if NAN_FOUND["first"] is not None:
                    return
                if _any_nan_in(output) and not _any_nan_in(args) and not _any_nan_in(kwargs):
                    NAN_FOUND["first"] = mod_name
                    print(f"\n*** FIRST NaN GENERATED IN: {mod_name} ***", flush=True)
                    print(f"    module type: {type(module).__name__}", flush=True)
                    if torch.is_tensor(output):
                        n_nan = torch.isnan(output).sum().item()
                        n_inf = torch.isinf(output).sum().item()
                        print(f"    output: shape={tuple(output.shape)} NaN={n_nan} Inf={n_inf}", flush=True)
                    print("    inputs were clean — this module is the source.", flush=True)
                    raise RuntimeError(f"NaN generated in {mod_name}")
            return hook

        try:
            mod.register_forward_hook(make_hook(full), with_kwargs=True)
        except TypeError:
            # Older PyTorch without with_kwargs support: fall back.
            def make_old_hook(mod_name):
                def hook(module, inp, output):
                    if NAN_FOUND["first"] is not None:
                        return
                    if _any_nan_in(output) and not _any_nan_in(inp):
                        NAN_FOUND["first"] = mod_name
                        print(f"\n*** FIRST NaN GENERATED IN: {mod_name} ***", flush=True)
                        raise RuntimeError(f"NaN generated in {mod_name}")
                return hook
            mod.register_forward_hook(make_old_hook(full))


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


def load_small_dpo_dataset(tokenizer, n=10):
    raw = load_dataset(DATASET_NAME, split="train").select(range(n))

    def msg_to_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("value", "")
        if isinstance(x, list):
            return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
        return str(x)

    def normalize(rec):
        prompt_raw = rec.get("prompt", "")
        if isinstance(prompt_raw, list):
            prompt = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            prompt = msg_to_text(prompt_raw).strip()
        return {
            "prompt": prompt,
            "chosen": msg_to_text(rec.get("chosen", "")).strip(),
            "rejected": msg_to_text(rec.get("rejected", "")).strip(),
        }

    return raw.map(normalize, remove_columns=raw.column_names)


def main():
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_small_dpo_dataset(tokenizer, n=10)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    )
    model.to(device)
    model.config.use_cache = False
    model.train()

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
        opt = SparseAdamW(
            model.named_parameters(), mask_manager=mm,
            lr=5e-7, block_size=16, mlp_only=True,
        )

        # Attach hooks to the policy. (TRL will deepcopy this for ref later.)
        attach_nan_hooks(model, "policy")
        print(f"  attached NaN hooks on {sum(1 for _ in model.modules())} policy modules")

        scratch_root = os.environ.get("SCRATCH_USER_ROOT", f"/scratch/{os.environ.get('USER', 'unknown')}")
        out_dir = os.path.join(scratch_root, "trl_nan_hook_out")
        os.makedirs(out_dir, exist_ok=True)

        cfg = DPOConfig(
            output_dir=out_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-7,
            max_steps=3,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
            beta=0.1,
            max_length=1024,
            max_prompt_length=512,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            save_strategy="no",
        )

        # Custom collator (same as sparse_DPO_timing_baseline.py).
        def collator(examples):
            prompts = [ex["prompt"] for ex in examples]
            chosens = [ex["chosen"] for ex in examples]
            rejects = [ex["rejected"] for ex in examples]
            bp = tokenizer(prompts, padding=True, truncation=True, max_length=512,
                           return_tensors="pt", pad_to_multiple_of=8)
            bc = tokenizer(chosens, padding=True, truncation=True, max_length=1024,
                           return_tensors="pt", pad_to_multiple_of=8)
            br = tokenizer(rejects, padding=True, truncation=True, max_length=1024,
                           return_tensors="pt", pad_to_multiple_of=8)
            return {
                "prompt_input_ids": bp["input_ids"].to(torch.long),
                "prompt_attention_mask": bp["attention_mask"].to(torch.long),
                "chosen_input_ids": bc["input_ids"].to(torch.long),
                "chosen_attention_mask": bc["attention_mask"].to(torch.long),
                "rejected_input_ids": br["input_ids"].to(torch.long),
                "rejected_attention_mask": br["attention_mask"].to(torch.long),
            }

        trainer = DPOTrainer(
            model=model, args=cfg, train_dataset=dataset,
            data_collator=collator, optimizers=(opt, None),
        )

        # If TRL clones model for reference, hook that too.
        if hasattr(trainer, "ref_model") and trainer.ref_model is not None:
            attach_nan_hooks(trainer.ref_model, "ref")
            print("  attached NaN hooks on ref_model")
        else:
            print("  no separate ref_model on trainer (PEFT-style implicit ref?)")

        print("\n=== START TRAIN (max 3 optim steps, hooks watching) ===\n")
        try:
            trainer.train()
            print("\n=== TRAIN DONE without NaN. Bug is more subtle than module-output NaN. ===")
        except RuntimeError as e:
            print(f"\n=== HALTED on NaN: {e} ===")
            print(f"Loss-history captured up to halt:")
            for entry in trainer.state.log_history[-5:]:
                print(f"  {entry}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()

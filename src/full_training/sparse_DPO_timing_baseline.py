#!/usr/bin/env python3
"""
Sparse BSR DPO Timing Baseline — Scott-style companion to DPO_timing_baseline.py.

Same config defaults (batch_size=1 × grad_accum=4, 10 steps, Light-R1 subset 10,
max_length=1024, max_prompt_length=512) and same timing methodology
(torch.cuda.Event + synchronize()) as DPO_timing_baseline.py. The delta:
  - Random block-sparse mask (default 90% sparsity, mlp-only)
  - nn.Linear → SparseLinearLayer via replace_linear_modules (BSR backward)
  - SparseAdamW optimizer (indexed gather/scatter step)

Output: sparse_baseline_timing.json (wall_time, gpu_time, per-step times).

Run: python sparse_DPO_timing_baseline.py --n_steps 10 --sparsity 0.9
"""

import os
import sys
import json
import time
import tempfile
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any

# Project root for `src.*` imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.mlps.bsr_sparse_mlp import replace_linear_modules
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "qihoo360/Light-R1-DPOData"
SUBSET_SIZE = 10


def load_dpo_dataset(subset_size=None):
    """Same Light-R1 loader as DPO_timing_baseline.py."""
    print(f"Loading dataset: {DATASET_NAME}")
    raw_ds = load_dataset(DATASET_NAME, split="train")

    def msg_to_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("value", "")
        if isinstance(x, list):
            return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
        return str(x)

    def normalize_record(rec):
        prompt_raw = rec.get("prompt", "")
        chosen_raw = rec.get("chosen", "")
        rejected_raw = rec.get("rejected", "")
        if isinstance(prompt_raw, list):
            prompt_text = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            prompt_text = msg_to_text(prompt_raw).strip()
        return {
            "prompt": prompt_text,
            "chosen": msg_to_text(chosen_raw).strip(),
            "rejected": msg_to_text(rejected_raw).strip(),
        }

    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    if subset_size is not None:
        norm_ds = norm_ds.select(range(min(subset_size, len(norm_ds))))
    print(f"✓ Loaded {len(norm_ds)} examples")
    return norm_ds


def dpo_collator_fn(examples, tokenizer):
    """DPO collator. Batch-tokenize each field once (avoids the per-example
    list-of-BatchEncodings → tokenizer.pad path that breaks on some inputs)."""
    prompts = [ex.get("prompt", "") for ex in examples]
    chosens = [ex.get("chosen", "") for ex in examples]
    rejects = [ex.get("rejected", "") for ex in examples]

    batch_prompt = tokenizer(
        prompts, padding=True, truncation=True, max_length=512,
        return_tensors="pt", pad_to_multiple_of=8,
    )
    batch_chosen = tokenizer(
        chosens, padding=True, truncation=True, max_length=1024,
        return_tensors="pt", pad_to_multiple_of=8,
    )
    batch_reject = tokenizer(
        rejects, padding=True, truncation=True, max_length=1024,
        return_tensors="pt", pad_to_multiple_of=8,
    )

    return {
        "prompt_input_ids": batch_prompt["input_ids"].to(torch.long),
        "prompt_attention_mask": batch_prompt["attention_mask"].to(torch.long),
        "chosen_input_ids": batch_chosen["input_ids"].to(torch.long),
        "chosen_attention_mask": batch_chosen["attention_mask"].to(torch.long),
        "rejected_input_ids": batch_reject["input_ids"].to(torch.long),
        "rejected_attention_mask": batch_reject["attention_mask"].to(torch.long),
    }


def build_random_block_mask_dict(
    model: nn.Module,
    sparsity: float,
    block_size: int,
    mlp_only: bool = True,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Random block-sparse bool masks per eligible weight matrix, on CPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    masks: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if mlp_only and "mlp" not in name.lower():
            continue
        dout, din = int(p.shape[0]), int(p.shape[1])
        nb_out = max(1, dout // block_size)
        nb_in = max(1, din // block_size)
        n_blocks = nb_out * nb_in
        n_keep = max(1, int(round((1.0 - sparsity) * n_blocks)))
        perm = torch.randperm(n_blocks, generator=g)[:n_keep]
        block_keep = torch.zeros(n_blocks, dtype=torch.bool)
        block_keep[perm] = True
        block_keep = block_keep.view(nb_out, nb_in)
        mask = block_keep.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        mask = mask[:dout, :din].contiguous()
        masks[name] = mask
    return masks


def train_sparse_baseline(
    n_steps=10,
    batch_size=1,
    learning_rate=5e-5,
    subset_size=10,
    sparsity=0.9,
    block_size_bsr=16,
    block_size_adam=128,
    mlp_only=True,
    disable_tf32=False,
):
    print(f"\n{'='*60}")
    print("SPARSE BSR DPO TIMING BASELINE")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Steps: {n_steps}  Batch: {batch_size}  Accum: 4  (effective batch 4)")
    print(f"Sparsity: {sparsity}  block_size_bsr: {block_size_bsr}  block_size_adam: {block_size_adam}  mlp_only: {mlp_only}")
    print(f"{'='*60}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading dataset...")
    dataset = load_dpo_dataset(subset_size=subset_size)

    def collator(examples):
        return dpo_collator_fn(examples, tokenizer)

    print(f"Loading model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(device)
    model.config.use_cache = False
    print(f"✓ Model on {device} dtype={model.dtype}\n")

    print(f"Building random block-sparse mask (sparsity={sparsity}, block={block_size_bsr}, mlp_only={mlp_only})...")
    masks = build_random_block_mask_dict(model, sparsity, block_size_bsr, mlp_only=mlp_only)
    tmp_fd, tmp_mask_path = tempfile.mkstemp(suffix=".pt", prefix="sparse_bsr_mask_")
    os.close(tmp_fd)
    torch.save(masks, tmp_mask_path)
    print(f"✓ Saved {len(masks)} masks to {tmp_mask_path}\n")

    try:
        mask_manager = SparseMaskManager(tmp_mask_path, device=device)

        print("Injecting SparseLinearLayer modules...")
        replace_linear_modules(
            model,
            mask_manager.masks,
            block_size=block_size_bsr,
            use_tf32=not disable_tf32,
            verbose=False,
        )
        print(f"✓ BSR injection complete\n")

        print("Building SparseAdamW optimizer...")
        optimizer = SparseAdamW(
            model.named_parameters(),
            mask_manager=mask_manager,
            lr=learning_rate,
            block_size=block_size_adam,
            mlp_only=mlp_only,
        )
        print("✓ SparseAdamW ready\n")

        # Output dir on /scratch + save_strategy="no" because Scott's
        # save_steps=n_steps+1 trick didn't actually stop TRL from writing
        # a checkpoint at end of training (we observed his run leave 15G of
        # model weights in ./baseline_temp/). Belt-and-suspenders here.
        scratch_root = os.environ.get("SCRATCH_USER_ROOT", f"/scratch/{os.environ.get('USER', 'unknown')}")
        out_dir = os.path.join(scratch_root, "sparse_baseline_temp")
        os.makedirs(out_dir, exist_ok=True)
        # logging_steps env override allows convergence checks to capture loss /
        # rewards/* / logps/* trajectories. Default is "disabled" (n_steps+1) so
        # pure timing runs aren't slowed by per-step logging hooks.
        log_steps_env = os.environ.get("RL_CASINO_LOGGING_STEPS")
        eff_logging_steps = int(log_steps_env) if log_steps_env else n_steps + 1
        dpo_config = DPOConfig(
            output_dir=out_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=n_steps,
            logging_steps=eff_logging_steps,
            report_to="none",
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            beta=0.1,
            max_length=1024,
            max_prompt_length=512,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=False,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            save_strategy="no",           # truly disable checkpointing
        )

        print("Initializing DPOTrainer...")
        # Use custom collator (with the batch-tokenize fix above). Without it,
        # DPOTrainer's default tokenization produced float input_ids that
        # crashed the embedding lookup with 'Expected Long, got Float'.
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            eval_dataset=None,
            data_collator=collator,
            optimizers=(optimizer, None),
        )
        print("✓ Trainer ready\n")

        print(f"{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}\n")

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.time()

        trainer.train()

        wall_time = time.time() - wall_start
        end_event.record()
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000.0

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Wall clock time: {wall_time:.2f}s")
        print(f"GPU time:        {gpu_time:.2f}s")
        print(f"Time per step (wall): {wall_time / n_steps:.2f}s")
        print(f"Time per step (gpu):  {gpu_time / n_steps:.2f}s")
        print(f"{'='*60}\n")

        results = {
            "method": "sparse_bsr",
            "wall_time": wall_time,
            "gpu_time": gpu_time,
            "time_per_step_wall": wall_time / n_steps,
            "time_per_step_gpu": gpu_time / n_steps,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "sparsity": sparsity,
            "block_size_bsr": block_size_bsr,
            "block_size_adam": block_size_adam,
            "mlp_only": mlp_only,
            "dtype": "bfloat16",
            "n_masks": len(masks),
        }
        out_path = "sparse_baseline_timing.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Timing results saved to {out_path}\n")

        # Save trainer.state.log_history if any logs were captured (convergence check).
        if trainer.state.log_history:
            log_path = "log_history.json"
            with open(log_path, "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(f"✓ log_history saved to {log_path}  ({len(trainer.state.log_history)} entries)\n")

    finally:
        if os.path.exists(tmp_mask_path):
            os.remove(tmp_mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse BSR DPO timing baseline (Scott-style)")
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--subset_size", type=int, default=10)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--block_size_bsr", type=int, default=16)
    parser.add_argument("--block_size_adam", type=int, default=128)
    parser.add_argument("--full_model", action="store_true",
                        help="Apply mask to all layers (default: mlp-only).")
    parser.add_argument("--disable_tf32", action="store_true",
                        help="Pass use_tf32=False to BSR kernel.")
    args = parser.parse_args()

    train_sparse_baseline(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset_size=args.subset_size,
        sparsity=args.sparsity,
        block_size_bsr=args.block_size_bsr,
        block_size_adam=args.block_size_adam,
        mlp_only=not args.full_model,
        disable_tf32=args.disable_tf32,
    )

#!/usr/bin/env python3
"""
Baseline Dense DPO Training - Timing Comparison Version

Stripped down version for clean performance comparison.
No logging, no checkpointing - just pure training time measurement.

Run: python baseline_dpo_timing.py --n_steps 10
"""

import os
import json
import time
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "qihoo360/Light-R1-DPOData"
SUBSET_SIZE = 10


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dpo_dataset(subset_size=None):
    """Load and normalize DPO dataset."""
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
                m.get("value","") for m in prompt_raw
                if isinstance(m, dict) and m.get("from","").lower() != "assistant"
            ).strip()
        else:
            prompt_text = msg_to_text(prompt_raw).strip()

        chosen_text = msg_to_text(chosen_raw).strip()
        rejected_text = msg_to_text(rejected_raw).strip()

        return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}

    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    
    if subset_size is not None:
        norm_ds = norm_ds.select(range(min(subset_size, len(norm_ds))))
    
    print(f"✓ Loaded {len(norm_ds)} examples")
    return norm_ds


def dpo_collator_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """Data collator for DPO training."""
    if "prompt_input_ids" in examples[0]:
        def pad_stack(key):
            seqs = [torch.tensor(ex[key]) if not torch.is_tensor(ex[key]) else ex[key] for ex in examples]
            lens = [s.size(-1) for s in seqs]
            maxlen = max(lens)
            out = torch.full((len(seqs), maxlen), fill_value=0, dtype=torch.long)
            mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.size(-1)] = s.to(torch.long)
                mask[i, : s.size(-1)] = 1
            return out, mask

        p_ids, p_mask = pad_stack("prompt_input_ids")
        c_ids, c_mask = pad_stack("chosen_input_ids")
        r_ids, r_mask = pad_stack("rejected_input_ids")
        return {
            "prompt_input_ids": p_ids, "prompt_attention_mask": p_mask,
            "chosen_input_ids": c_ids, "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids, "rejected_attention_mask": r_mask,
        }

    prompts = [ex.get("prompt", "") for ex in examples]
    chosens = [ex.get("chosen", "") for ex in examples]
    rejects = [ex.get("rejected", "") for ex in examples]

    enc_prompt = [tokenizer(p, truncation=True, max_length=512, return_tensors="pt") for p in prompts]
    enc_chosen = [tokenizer(c, truncation=True, max_length=1024, return_tensors="pt") for c in chosens]
    enc_reject = [tokenizer(r, truncation=True, max_length=1024, return_tensors="pt") for r in rejects]

    batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)
    batch_chosen = tokenizer.pad(enc_chosen, padding=True, return_tensors="pt", pad_to_multiple_of=8)
    batch_reject = tokenizer.pad(enc_reject, padding=True, return_tensors="pt", pad_to_multiple_of=8)

    for k in ("input_ids", "attention_mask"):
        batch_prompt[k] = batch_prompt[k].to(torch.long)
        batch_chosen[k] = batch_chosen[k].to(torch.long)
        batch_reject[k] = batch_reject[k].to(torch.long)

    return {
        "prompt_input_ids": batch_prompt["input_ids"],
        "prompt_attention_mask": batch_prompt["attention_mask"],
        "chosen_input_ids": batch_chosen["input_ids"],
        "chosen_attention_mask": batch_chosen["attention_mask"],
        "rejected_input_ids": batch_reject["input_ids"],
        "rejected_attention_mask": batch_reject["attention_mask"],
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_baseline(
    checkpoint_path=None,
    n_steps=10,
    batch_size=1,
    learning_rate=5e-5,
    subset_size=10,
):
    """Baseline dense DPO training with timing measurement."""
    
    model_path = checkpoint_path if checkpoint_path else MODEL_NAME
    
    print(f"\n{'='*60}")
    print(f"BASELINE DENSE DPO TRAINING")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Steps: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dpo_dataset(subset_size=subset_size)
    print("✓ Dataset loaded\n")
    
    def collator(examples):
        return dpo_collator_fn(examples, tokenizer)

    # Load model
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not torch.cuda.is_available() or model.device.type == 'cpu':
        model.to(device)
    
    model.config.use_cache = False
    print(f"✓ Model loaded on {device} with dtype: {model.dtype}\n")

    # Configure DPO
    dpo_config = DPOConfig(
        output_dir="./baseline_temp",
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=n_steps,
        logging_steps=n_steps + 1,  # Disable logging
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        save_steps=n_steps + 1,  # Disable saving
    )

    # Initialize trainer
    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
    )
    print("✓ Trainer ready\n")

    # Train with timing
    print(f"{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    # Use CUDA events for GPU timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    wall_start = time.time()
    trainer.train()
    wall_time = time.time() - wall_start
    
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000.0
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Wall clock time: {wall_time:.2f}s")
    if torch.cuda.is_available():
        print(f"GPU time: {gpu_time:.2f}s")
        print(f"Time per step (GPU): {gpu_time/n_steps:.2f}s")
    print(f"Time per step (wall): {wall_time/n_steps:.2f}s")
    print(f"{'='*60}\n")
    
    # Save timing results
    results = {
        'wall_time': wall_time,
        'time_per_step_wall': wall_time / n_steps,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    
    if torch.cuda.is_available():
        results['gpu_time'] = gpu_time
        results['time_per_step_gpu'] = gpu_time / n_steps
    
    with open('baseline_timing.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Timing results saved to baseline_timing.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline dense DPO training for timing comparison")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (default: None = use base model)")
    parser.add_argument("--n_steps", type=int, default=10,
                       help="Number of training steps (default: 10)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--subset_size", type=int, default=10,
                       help="Dataset subset size (default: 10)")
    
    args = parser.parse_args()
    
    train_baseline(
        checkpoint_path=args.checkpoint,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset_size=args.subset_size,
    )
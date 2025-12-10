#!/usr/bin/env python3
"""
Baseline Dense GRPO Training - Timing Comparison Version

Stripped down version for clean performance comparison.
No logging, no checkpointing - just pure training time measurement.

Run: python GRPO_timing_baseline.py --n_steps 10
"""

import os
import json
import time
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from typing import List, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "open-r1/OpenR1-Math-220k"
SUBSET_SIZE = 10


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_grpo_dataset(subset_size=None):
    """Load and normalize OpenR1 dataset for GRPO."""
    print(f"Loading dataset: {DATASET_NAME}")
    raw_ds = load_dataset(DATASET_NAME, split="default")
    
    def normalize_record(rec):
        """
        Normalize OpenR1 dataset record to format expected by GRPO.
        OpenR1 has 'prompt' and 'solution' fields.
        """
        prompt_raw = rec.get("prompt", "")
        solution_raw = rec.get("solution", "")
        
        # Handle prompt - could be string or list
        if isinstance(prompt_raw, list):
            prompt_text = "\n".join(
                m.get("value", "") if isinstance(m, dict) else str(m)
                for m in prompt_raw
            ).strip()
        elif isinstance(prompt_raw, dict):
            prompt_text = prompt_raw.get("value", str(prompt_raw))
        else:
            prompt_text = str(prompt_raw).strip()
        
        # Handle solution - could be string or list
        if isinstance(solution_raw, list):
            solution_text = "\n".join(
                m.get("value", "") if isinstance(m, dict) else str(m)
                for m in solution_raw
            ).strip()
        elif isinstance(solution_raw, dict):
            solution_text = solution_raw.get("value", str(solution_raw))
        else:
            solution_text = str(solution_raw).strip()
        
        return {"prompt": prompt_text, "solution": solution_text}

    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    
    if subset_size is not None:
        norm_ds = norm_ds.select(range(min(subset_size, len(norm_ds))))
    
    print(f"✓ Loaded {len(norm_ds)} examples")
    return norm_ds


def grpo_collator_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """Data collator for GRPO training."""
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
        return {
            "prompt_input_ids": p_ids,
            "prompt_attention_mask": p_mask,
        }

    prompts = [ex.get("prompt", "") for ex in examples]

    enc_prompt = [tokenizer(p, truncation=True, max_length=512, return_tensors="pt") for p in prompts]
    batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)

    for k in ("input_ids", "attention_mask"):
        batch_prompt[k] = batch_prompt[k].to(torch.long)

    return {
        "prompt_input_ids": batch_prompt["input_ids"],
        "prompt_attention_mask": batch_prompt["attention_mask"],
    }


# ============================================================================
# REWARD FUNCTION
# ============================================================================

def reward_function(completions: List[str] | List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Reward function for GRPO training.
    
    Naive baseline reward function - same as GRPO_train.py.
    """
    rewards = []
    for completion in completions:
        # Handle conversational format (list of dicts)
        if isinstance(completion, list):
            # Extract text from conversational format
            completion_text = " ".join(
                msg.get("content", "") if isinstance(msg, dict) else str(msg)
                for msg in completion
            )
        else:
            completion_text = str(completion)
        
        # Simple reward: encourage longer completions with reasoning
        base_reward = len(completion_text.split()) / 100.0  # Normalize by word count
        
        # Bonus for having solution markers (if dataset uses them)
        if "<SOLUTION>" in completion_text or "</SOLUTION>" in completion_text:
            base_reward += 0.5
        
        # Bonus for having reasoning markers
        if "<think>" in completion_text.lower() or "<start_working_out>" in completion_text.lower():
            base_reward += 0.3
        
        rewards.append(min(base_reward, 1.0))  # Cap at 1.0
    
    return rewards


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
    """Baseline dense GRPO training with timing measurement."""
    
    model_path = checkpoint_path if checkpoint_path else MODEL_NAME
    
    print(f"\n{'='*60}")
    print(f"BASELINE DENSE GRPO TRAINING")
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
    dataset = load_grpo_dataset(subset_size=subset_size)
    print("✓ Dataset loaded\n")
    
    def collator(examples):
        return grpo_collator_fn(examples, tokenizer)

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

    # Configure GRPO
    grpo_config = GRPOConfig(
        output_dir="./baseline_temp",
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=n_steps,
        logging_steps=n_steps + 1,  # Disable logging
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        beta=0.1,  # KL penalty coefficient
        max_length=1024,
        max_prompt_length=512,
        num_generations=8,  # Number of generations per prompt
        generation_batch_size=8,  # Batch size for generation
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        save_steps=n_steps + 1,  # Disable saving
    )

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        eval_dataset=None,
        reward_funcs=[reward_function],
        data_collator=collator,
        processing_class=tokenizer,
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
    
    with open('baseline_grpo_timing.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Timing results saved to baseline_grpo_timing.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline dense GRPO training for timing comparison")
    
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

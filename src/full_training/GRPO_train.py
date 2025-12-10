import os
import json
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
)
from trl import GRPOTrainer, GRPOConfig
from typing import List, Dict, Any


#######################################
# 0. Config
#######################################

def sanitize_model_name(model_name: str) -> str:
    """
    Convert HuggingFace model name to filesystem-safe string.
    
    Examples:
        "google/gemma-3-270m-it" -> "google_gemma_3_270m_it"
        "meta-llama/Llama-3.1-8B" -> "meta_llama_llama_3_1_8b"
    """
    # Replace "/" with "_", replace "-" with "_", convert to lowercase
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    # Remove any remaining special characters that might cause issues
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    # Collapse multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


parser = argparse.ArgumentParser(description="GRPO Training Script")
parser.add_argument(
    "--model_name",
    type=str,
    default="google/gemma-3-270m-it",
    help="HuggingFace model name to load (default: google/gemma-3-270m-it)"
)
args = parser.parse_args()

MODEL_NAME = args.model_name
MODEL_NAME_SANITIZED = sanitize_model_name(MODEL_NAME)
DATASET_NAME = "open-r1/OpenR1-Math-220k"  # OpenR1 dataset
OUTPUT_DIR = f"./checkpoints_{MODEL_NAME_SANITIZED}_grpo"
DELTA_LOG_DIR = f"./delta_logs_{MODEL_NAME_SANITIZED}_grpo"

# Flexible checkpoint schedule
# Save every 5 steps for first 25 steps, then every 25 steps after
# CHECKPOINT_SCHEDULE = (
#     list(range(5, 25, 5)) +  # [5, 10, 15, 20, 25]
#     list(range(50, 101, 25))  # [50, 75, 100]
# )

CHECKPOINT_SCHEDULE = (
    list(range(10, 90, 10)) +  # [10, 20, 30, 40, 50, 60, 70, 80, 90]
    list(range(100, 1001, 100))  # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
)
# [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

THRESHOLD = 1e-3
NUM_STEPS = 1000
SUBSET_SIZE = None  # reduce for faster bring-up

WANDB_PROJECT = f"{MODEL_NAME_SANITIZED}-grpo-subnetwork-emergence"
WANDB_RUN_NAME = f"{MODEL_NAME_SANITIZED}_openr1_flexible_checkpoints"

print(f"Checkpoint schedule: {CHECKPOINT_SCHEDULE}")

#######################################
# 1. Load dataset
#######################################

raw_ds = load_dataset(DATASET_NAME, split="default")  # Use default split

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

# (Optional) take a subset to iterate quickly
if SUBSET_SIZE is not None:
    norm_ds = norm_ds.select(range(min(SUBSET_SIZE, len(norm_ds))))

train_dataset = norm_ds
eval_dataset = None

#######################################
# 2. Load tokenizer
#######################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def grpo_collator_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collator function for GRPO training.
    GRPO expects prompts and generates completions, so we only need to tokenize prompts.
    """
    # If the dataset was already preprocessed, just stack/pad those tensors.
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

    # Otherwise, we expect raw strings.
    prompts = [ex.get("prompt", "") for ex in examples]

    enc_prompt = [tokenizer(p, truncation=True, max_length=512, return_tensors="pt") for p in prompts]
    batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)

    for k in ("input_ids", "attention_mask"):
        batch_prompt[k] = batch_prompt[k].to(torch.long)

    return {
        "prompt_input_ids": batch_prompt["input_ids"],
        "prompt_attention_mask": batch_prompt["attention_mask"],
    }


#######################################
# 3. Reward function for GRPO

# TODO: Replace with proper reward function, not this dumb-ahh placeholder XD 
#######################################

def reward_function(completions: List[str] | List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Reward function for GRPO training.
    
    For OpenR1, we can use a simple reward based on:
    - Length of completion (encourage longer reasoning)
    - Presence of solution markers (if present in dataset)
    - Or use a more sophisticated reward model
    
    This is a simple placeholder - you may want to replace with a proper reward model.
    
    Args:
        completions: List of strings or list of conversational messages
        **kwargs: Additional dataset columns (e.g., ground_truth, solution)
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
        # You can replace this with a proper reward model
        base_reward = len(completion_text.split()) / 100.0  # Normalize by word count
        
        # Bonus for having solution markers (if dataset uses them)
        if "<SOLUTION>" in completion_text or "</SOLUTION>" in completion_text:
            base_reward += 0.5
        
        # Bonus for having reasoning markers
        if "<think>" in completion_text.lower() or "<start_working_out>" in completion_text.lower():
            base_reward += 0.3
        
        # Check if ground truth solution is provided in kwargs
        if "solution" in kwargs:
            # Could add reward based on matching solution (simplified here)
            pass
        
        rewards.append(min(base_reward, 1.0))  # Cap at 1.0
    
    return rewards


#######################################
# 4. Load policy model (trainable)
#######################################

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False  # Trainer compat

#######################################
# 5. GRPOConfig
#######################################

cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name=WANDB_RUN_NAME,
    report_to=["wandb"],
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_steps=NUM_STEPS,
    num_train_epochs=1,
    bf16=True,
    fp16=False,
    logging_steps=1,
    save_steps=999999,  # Disabled - we'll handle saving in callback
    save_total_limit=None,
    remove_unused_columns=False,
    # GRPO-specific knobs:
    num_generations=8,  # Number of generations per prompt for group-based advantage estimation
    generation_batch_size=8,  # Batch size for generation
    max_length=1024,
    max_prompt_length=512,
    beta=0.1,  # KL penalty coefficient (controls deviation from reference policy)
)

trainer = GRPOTrainer(
    model=model,
    args=cfg,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_funcs=[reward_function],  # GRPOTrainer expects a list of reward functions
    data_collator=grpo_collator_fn,
    processing_class=tokenizer,
)

#######################################
# 6. Snapshot initial params θ(0)
#######################################

base_state = {}
with torch.no_grad():
    for name, param in trainer.model.named_parameters():
        base_state[name] = param.detach().float().cpu().clone()

os.makedirs(DELTA_LOG_DIR, exist_ok=True)
torch.save(base_state, os.path.join(DELTA_LOG_DIR, "base_state.pt"))

#######################################
# 7. Flexible Checkpoint Callback
#######################################

class FlexibleCheckpointCallback(TrainerCallback):
    """
    Callback that saves deltas on a flexible schedule and tracks statistics.
    
    Schedule: Every 5 steps for first 25 steps, then every 25 steps after.
    Identical to DPO_train.py callback.
    """
    
    def __init__(self, base_state, delta_log_dir, checkpoint_schedule, threshold):
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.checkpoint_schedule = set(checkpoint_schedule)  # Use set for O(1) lookup
        self.threshold = threshold
        os.makedirs(self.delta_log_dir, exist_ok=True)
        self.wandb_initialized = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.wandb_initialized:
            wandb.init(
                project=WANDB_PROJECT,
                name=args.run_name,
                config={
                    "model_name": MODEL_NAME,
                    "dataset": DATASET_NAME,
                    "subset_size": SUBSET_SIZE,
                    "learning_rate": args.learning_rate,
                    "batch_size_per_device": args.per_device_train_batch_size,
                    "grad_accum": args.gradient_accumulation_steps,
                    "checkpoint_schedule": sorted(list(self.checkpoint_schedule)),
                    "num_generations": args.num_generations if hasattr(args, 'num_generations') else None,
                    "generation_batch_size": args.generation_batch_size if hasattr(args, 'generation_batch_size') else None,
                },
            )
            self.wandb_initialized = True

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        step = state.global_step

        layer_stats = {}
        full_deltas_to_save = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                current = param.detach().float().cpu()
                diff = current - self.base_state[name]

                l2 = torch.norm(diff).item()
                frac_big = (diff.abs() > self.threshold).float().mean().item()

                layer_stats[name] = {
                    "l2_from_init": l2,
                    "frac_big_from_init": frac_big,
                }

                # Check if this step is in our checkpoint schedule
                if step in self.checkpoint_schedule:
                    full_deltas_to_save[name] = diff.clone()

        # Always save stats (cheap JSON file)
        stats_path = os.path.join(self.delta_log_dir, f"stats_step_{step}.json")
        with open(stats_path, "w") as f:
            json.dump(layer_stats, f)

        # Aggregate summaries for wandb
        all_l2 = [v["l2_from_init"] for v in layer_stats.values()]
        all_frac = [v["frac_big_from_init"] for v in layer_stats.values()]
        mean_l2 = sum(all_l2) / len(all_l2)
        mean_frac = sum(all_frac) / len(all_frac)

        attn_l2 = []
        mlp_l2 = []
        for n, st in layer_stats.items():
            low = n.lower()
            if "attn" in low or "q_proj" in low or "k_proj" in low or "v_proj" in low or "o_proj" in low:
                attn_l2.append(st["l2_from_init"])
            if "mlp" in low or "ffn" in low or "feed_forward" in low or "gate_proj" in low or "up_proj" in low or "down_proj" in low:
                mlp_l2.append(st["l2_from_init"])

        wandb.log({
            "step": step,
            "subnet/mean_l2_from_init": mean_l2,
            "subnet/mean_frac_big_from_init": mean_frac,
            "subnet/attn_mean_l2": (sum(attn_l2)/len(attn_l2)) if attn_l2 else 0.0,
            "subnet/mlp_mean_l2": (sum(mlp_l2)/len(mlp_l2)) if mlp_l2 else 0.0,
        }, step=step)

        # Save full deltas on checkpoint schedule
        if step in self.checkpoint_schedule:
            delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
            torch.save(full_deltas_to_save, delta_file)
            print(f"  ✓ Saved checkpoint at step {step}")
            # Note: Not uploading to wandb since artifacts are already saved on cloud GPU

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.wandb_initialized:
            wandb.finish()

#######################################
# 8. Register callback and train
#######################################

trainer.add_callback(
    FlexibleCheckpointCallback(
        base_state=base_state,
        delta_log_dir=DELTA_LOG_DIR,
        checkpoint_schedule=CHECKPOINT_SCHEDULE,
        threshold=THRESHOLD,
    )
)

print(f"\n{'='*60}")
print(f"Starting GRPO training with flexible checkpoint schedule")
print(f"{'='*60}")
print(f"Checkpoints will be saved at steps: {CHECKPOINT_SCHEDULE}")
print(f"Total steps: {NUM_STEPS}")
print(f"{'='*60}\n")

trainer.train()

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"{'='*60}")
print(f"Deltas saved to: {DELTA_LOG_DIR}")
print(f"Model checkpoints saved to: {OUTPUT_DIR}")
print(f"{'='*60}\n")

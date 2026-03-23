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
from trl import DPOTrainer, DPOConfig
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


parser = argparse.ArgumentParser(description="DPO Training Script")
parser.add_argument(
    "--model_name",
    type=str,
    default="google/gemma-3-270m-it",
    help="HuggingFace model name to load (default: google/gemma-3-270m-it)"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="light-r1",
    help="Dataset key from registry (light-r1, tulu3, math-step-dpo, codepref) or HuggingFace ID"
)
parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB")
parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
parser.add_argument("--num_steps", type=int, default=500, help="Number of training steps (default: 500)")
parser.add_argument("--subset_size", type=int, default=None, help="Limit dataset size (default: None = full)")
parser.add_argument("--output_base_dir", type=str, default="/scratch/biggs.s/rl_casino_outputs", help="Base directory for all outputs (checkpoints, deltas)")
parser.add_argument("--dataset_cache_dir", type=str, default="/scratch/biggs.s/hf_cache/datasets", help="Cache directory for HuggingFace datasets")
args = parser.parse_args()

# Set dataset cache directory
os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir


MODEL_NAME = args.model_name
MODEL_NAME_SANITIZED = sanitize_model_name(MODEL_NAME)

# Dataset resolution via registry
from src.utils.dataset_registry import get_dataset_config, sanitize_dataset_name
DATASET_KEY = args.dataset
DATASET_CONFIG = get_dataset_config(DATASET_KEY)
DATASET_NAME = DATASET_CONFIG["hf_id"]
DATASET_SANITIZED = DATASET_CONFIG["sanitized_name"]

# Combined output path on scratch
BASE_DIR = args.output_base_dir
SUB_DIR = f"{MODEL_NAME_SANITIZED}_{DATASET_SANITIZED}"

OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints", SUB_DIR)
DELTA_LOG_DIR = os.path.join(BASE_DIR, "deltas", SUB_DIR)

# Flexible checkpoint schedule
# Save every 5 steps for first 25 steps, then every 25 steps after
# CHECKPOINT_SCHEDULE = (
#     list(range(5, 25, 5)) +  # [5, 10, 15, 20, 25]
#     list(range(50, 101, 25))  # [50, 75, 100]
# )

CHECKPOINT_SCHEDULE = (
    list(range(10, 51, 10)) +  # [10, 20, 30, 40, 50]
    list(range(100, 501, 100))  # [100, 200, 300, 400, 500]
)


THRESHOLD = 1e-5 # appendix A in Mukherjee et al 2025
NUM_STEPS = args.num_steps
SUBSET_SIZE = args.subset_size

WANDB_PROJECT = "huggingface"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
WANDB_RUN_NAME = args.run_name if args.run_name else f"{MODEL_NAME_SANITIZED}_{DATASET_SANITIZED}_dpo_{NUM_STEPS}steps"

print(f"Checkpoint schedule: {CHECKPOINT_SCHEDULE}")

#######################################
# 1. Load dataset
#######################################

from src.utils.dataset_registry import load_dpo_dataset as registry_load_dpo
from src.utils.data_utils import dpo_collator_fn

raw_ds = registry_load_dpo(DATASET_KEY, subset_size=SUBSET_SIZE)
train_dataset = raw_ds
eval_dataset = None

#######################################
# 2. Load tokenizer
#######################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def dpo_collator_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
        c_ids, c_mask = pad_stack("chosen_input_ids")
        r_ids, r_mask = pad_stack("rejected_input_ids")
        return {
            "prompt_input_ids": p_ids, "prompt_attention_mask": p_mask,
            "chosen_input_ids": c_ids, "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids, "rejected_attention_mask": r_mask,
        }

    # Otherwise, we expect raw strings.
    prompts  = [ex.get("prompt", "")   for ex in examples]
    chosens  = [ex.get("chosen", "")   for ex in examples]
    rejects  = [ex.get("rejected", "") for ex in examples]

    enc_prompt = [tokenizer(p, truncation=True, max_length=512,  return_tensors="pt") for p in prompts]
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
        "prompt_input_ids":        batch_prompt["input_ids"],
        "prompt_attention_mask":   batch_prompt["attention_mask"],
        "chosen_input_ids":        batch_chosen["input_ids"],
        "chosen_attention_mask":   batch_chosen["attention_mask"],
        "rejected_input_ids":      batch_reject["input_ids"],
        "rejected_attention_mask": batch_reject["attention_mask"],
    }

#######################################
# 3. Load policy model (trainable)
#######################################

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False  # Trainer compat

#######################################
# 4. DPOConfig
#######################################

cfg = DPOConfig(
    output_dir=OUTPUT_DIR,
    run_name=WANDB_RUN_NAME,
    report_to=["wandb"] if args.use_wandb else [],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_steps=NUM_STEPS,
    num_train_epochs=1,
    bf16=True, 
    fp16=False,
    logging_steps=1,
    save_steps=999999,  # Disabled - we'll handle saving in callback
    save_total_limit=None,
    remove_unused_columns=False,
    # DPO knobs:
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=cfg,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=dpo_collator_fn,
)

#######################################
# 5. Snapshot initial params θ(0)
#######################################

base_state = {}
with torch.no_grad():
    for name, param in trainer.model.named_parameters():
        base_state[name] = param.detach().float().cpu().clone()

os.makedirs(DELTA_LOG_DIR, exist_ok=True)
torch.save(base_state, os.path.join(DELTA_LOG_DIR, "base_state.pt"))

#######################################
# 6. Flexible Checkpoint Callback
#######################################

class FlexibleCheckpointCallback(TrainerCallback):
    """
    Callback that saves deltas on a flexible schedule and tracks statistics.
    
    Schedule: Every 5 steps for first 25 steps, then every 25 steps after.
    """
    
    def __init__(self, base_state, delta_log_dir, checkpoint_schedule, threshold):
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.checkpoint_schedule = set(checkpoint_schedule)  # Use set for O(1) lookup
        self.threshold = threshold
        os.makedirs(self.delta_log_dir, exist_ok=True)
        self.wandb_initialized = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.wandb_initialized and "wandb" in args.report_to:
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
# 7. Register callback and train
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
print(f"Starting DPO training with flexible checkpoint schedule")
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
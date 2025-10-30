import os
import json
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import DPODataCollatorWithPadding

#######################################
# 0. Config
#######################################

MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "qihoo360/Light-R1-DPOData"  # preference dataset (prompt/chosen/rejected) :contentReference[oaicite:4]{index=4}
OUTPUT_DIR = "./checkpoints_gemma3_dpo"
DELTA_LOG_DIR = "./delta_logs"
FULL_DUMP_EVERY = 200
THRESHOLD = 1e-4
SUBSET_SIZE = 500  # reduce for faster bring-up

WANDB_PROJECT = "gemma3-dpo-subnetwork-emergence"
WANDB_RUN_NAME = "gemma3-270m-it_lightR1_subset"

#######################################
# 1. Load dataset
#######################################

raw_ds = load_dataset("qihoo360/Light-R1-DPOData", split="train")

def msg_to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("value", "")
    if isinstance(x, list):
        return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
    return str(x)

def normalize_record(rec):
    prompt_raw   = rec.get("prompt", "")
    chosen_raw   = rec.get("chosen", "")
    rejected_raw = rec.get("rejected", "")

    if isinstance(prompt_raw, list):
        prompt_text = "\n".join(
            m.get("value","") for m in prompt_raw
            if isinstance(m, dict) and m.get("from","").lower() != "assistant"
        ).strip()
    else:
        prompt_text = msg_to_text(prompt_raw).strip()

    chosen_text   = msg_to_text(chosen_raw).strip()
    rejected_text = msg_to_text(rejected_raw).strip()

    return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}

raw_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)

# (Optional) take a subset to iterate quickly
if SUBSET_SIZE is not None:
    norm_ds = norm_ds.select(range(min(SUBSET_SIZE, len(norm_ds))))

train_dataset = norm_ds
eval_dataset = None

data_collator = DPODataCollatorWithPadding(
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
    pad_to_multiple_of=8,   # optional, helps Tensor Cores
)

#######################################
# 2. Load tokenizer
#######################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#######################################
# 3. Load policy model (trainable) + ref model (frozen)
#######################################

# policy model: this one will get updated
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False  # Trainer compat

#######################################
# 4. TrainingArguments
#######################################

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    bf16=True,
    fp16=False,
    logging_steps=1,
    report_to=["wandb"],
    run_name=WANDB_RUN_NAME,
    save_steps=FULL_DUMP_EVERY,
    save_total_limit=5,
)

#######################################
# 5. Create the DPOTrainer
#######################################

cfg = DPOConfig(
    output_dir=OUTPUT_DIR,
    run_name=WANDB_RUN_NAME,
    report_to=["wandb"],
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    bf16=True, fp16=False,
    logging_steps=1,
    save_steps=FULL_DUMP_EVERY,
    save_total_limit=5,
    # DPO knobs:
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=cfg,
    processing_class=tokenizer,   # ok to keep; collator will do the heavy lifting
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # <<< THIS ensures int64 input_ids
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
# 7. Callback: wandb stats + periodic full deltas
#######################################

class DeltaTrackingCallback(TrainerCallback):
    def __init__(self, base_state, delta_log_dir, full_dump_every, threshold):
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.full_dump_every = full_dump_every
        self.threshold = threshold
        os.makedirs(self.delta_log_dir, exist_ok=True)
        self.wandb_initialized = False

    def on_train_begin(self, args, state, control, **kwargs):
        # initialize wandb run manually for custom logging
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

                # collect full deltas for periodic artifact dumps
                if step % self.full_dump_every == 0 and step > 0:
                    full_deltas_to_save[name] = diff.clone()

        # write a local JSON backup of stats
        stats_path = os.path.join(self.delta_log_dir, f"stats_step_{step}.json")
        with open(stats_path, "w") as f:
            json.dump(layer_stats, f)

        # aggregate a few nice summaries for wandb
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

        # every N steps, dump full deltas and upload artifact
        if step % self.full_dump_every == 0 and step > 0:
            delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
            torch.save(full_deltas_to_save, delta_file)

            artifact = wandb.Artifact(
                name=f"deltas_step_{step}",
                type="model-deltas",
                metadata={"global_step": step}
            )
            artifact.add_file(delta_file)
            wandb.log_artifact(artifact)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.wandb_initialized:
            wandb.finish()

#######################################
# 8. Register callback and train
#######################################

trainer.add_callback(
    DeltaTrackingCallback(
        base_state=base_state,
        delta_log_dir=DELTA_LOG_DIR,
        full_dump_every=FULL_DUMP_EVERY,
        threshold=THRESHOLD,
    )
)

trainer.train()
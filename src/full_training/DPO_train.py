import os
import json
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from trl import DPOTrainer

#######################################
# 0. Config
#######################################

MODEL_NAME = "google/gemma-3-270m-it"          # base model
DATASET_NAME = "qihoo360/Light-R1-DPOData"     # DPO pairs dataset  (Apache-2.0)  :contentReference[oaicite:1]{index=1}
OUTPUT_DIR = "./checkpoints_gemma3_dpo"
DELTA_LOG_DIR = "./delta_logs"
FULL_DUMP_EVERY = 200          # steps: how often we save full deltas to disk + wandb artifact
THRESHOLD = 1e-4               # "moved" cutoff for frac_big_from_init
SUBSET_SIZE = 500              # take first N examples for initial run; set None for full

WANDB_PROJECT = "gemma3-dpo-subnetwork-emergence"
WANDB_RUN_NAME = "gemma3-270m-it_lightR1_subset"


#######################################
# 1. Load dataset (prompt/chosen/rejected)
#######################################

# The Light-R1-DPOData repo stores DPO-style pairs used for preference optimization
# We assume it's a single split "train" with "prompt", "chosen", "rejected".
raw_ds = load_dataset(DATASET_NAME, split="train")  # :contentReference[oaicite:2]{index=2}

if SUBSET_SIZE is not None:
    raw_ds = raw_ds.select(range(min(SUBSET_SIZE, len(raw_ds))))

# TRL's DPOTrainer expects a dataset that yields dicts with these exact keys.
train_dataset = raw_ds
eval_dataset = None  # you can also do a small held-out slice


#######################################
# 2. Load model + tokenizer
#######################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Gemma-style tokenizers typically define eos; make sure there's a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # bf16 if GPU supports it; else use float16
    device_map="auto",
)
model.config.use_cache = False  # important for HF Trainer training loops


#######################################
# 3. Training arguments
#######################################

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    # effective batch size = 16 sequences/step here (2 * 8), adjust as needed

    learning_rate=5e-6,
    num_train_epochs=1,             # can bump later
    max_steps=None,                 # or set an explicit step budget

    bf16=True,                      # match model dtype if possible
    fp16=False,                     # don't mix both fp16+bf16
    logging_steps=1,                # we want dense logging
    report_to=["wandb"],            # HF trainer will log to wandb
    run_name=WANDB_RUN_NAME,

    save_steps=FULL_DUMP_EVERY,     # normal HF checkpoint cadence
    save_total_limit=5,
)

#######################################
# 4. Create DPO trainer
#######################################

trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,                       # common DPO beta for preference sharpness
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
)

#######################################
# 5. Snapshot initial weights θ(0) in fp32 on CPU
#######################################

base_state = {}
with torch.no_grad():
    for name, param in trainer.model.named_parameters():
        base_state[name] = param.detach().float().cpu().clone()

os.makedirs(DELTA_LOG_DIR, exist_ok=True)
torch.save(base_state, os.path.join(DELTA_LOG_DIR, "base_state.pt"))


#######################################
# 6. Define callback to:
#    - compute layer drift stats each step
#    - log stats to wandb
#    - dump full deltas periodically AND upload them as wandb artifacts
#######################################

class DeltaTrackingCallback(TrainerCallback):
    def __init__(self, base_state, delta_log_dir, full_dump_every, threshold):
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.full_dump_every = full_dump_every
        self.threshold = threshold

        os.makedirs(self.delta_log_dir, exist_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        # init wandb manually so we can also log custom stuff
        wandb.init(project=WANDB_PROJECT, name=args.run_name, config={
            "model_name": MODEL_NAME,
            "dataset": DATASET_NAME,
            "subset_size": SUBSET_SIZE,
            "beta": kwargs["trainer"].beta if hasattr(kwargs["trainer"], "beta") else None,
            "learning_rate": args.learning_rate,
            "batch_size_per_device": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
        })

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        step = state.global_step

        layer_stats = {}
        full_deltas_to_save = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                current = param.detach().float().cpu()
                diff = current - self.base_state[name]

                # stats
                l2 = torch.norm(diff).item()
                frac_big = (diff.abs() > self.threshold).float().mean().item()

                layer_stats[name] = {
                    "l2_from_init": l2,
                    "frac_big_from_init": frac_big,
                }

                # capture full tensor deltas every N steps
                if step % self.full_dump_every == 0 and step > 0:
                    full_deltas_to_save[name] = diff.clone()

        # --- 6a. write per-step stats json (local, mostly for backup) ---
        stats_path = os.path.join(self.delta_log_dir, f"stats_step_{step}.json")
        with open(stats_path, "w") as f:
            json.dump(layer_stats, f)

        # --- 6b. log scalar summaries to wandb ---
        # We don't want to spam wandb with thousands of keys per step for giant models,
        # so we'll aggregate a few interpretable summaries across layers.
        # Example summaries:
        #   - mean L2 across all tensors
        #   - top-k movers (attention_out, mlp_up, etc.) can be added later

        all_l2 = [v["l2_from_init"] for v in layer_stats.values()]
        all_frac = [v["frac_big_from_init"] for v in layer_stats.values()]
        mean_l2 = sum(all_l2) / len(all_l2)
        mean_frac = sum(all_frac) / len(all_frac)

        # Also log a couple specific interesting modules by name pattern:
        # e.g. attention vs MLP drift
        attn_l2 = []
        mlp_l2 = []
        for name, st in layer_stats.items():
            if "attn" in name.lower() or "q_proj" in name.lower() or "k_proj" in name.lower() or "v_proj" in name.lower():
                attn_l2.append(st["l2_from_init"])
            if "mlp" in name.lower() or "feed_forward" in name.lower() or "gate_proj" in name.lower() or "up_proj" in name.lower():
                mlp_l2.append(st["l2_from_init"])

        wandb.log({
            "step": step,
            "subnet/mean_l2_from_init": mean_l2,
            "subnet/mean_frac_big_from_init": mean_frac,
            "subnet/attn_mean_l2": (sum(attn_l2)/len(attn_l2)) if attn_l2 else 0.0,
            "subnet/mlp_mean_l2": (sum(mlp_l2)/len(mlp_l2)) if mlp_l2 else 0.0,
        }, step=step)

        # --- 6c. dump full deltas and push as W&B artifact occasionally ---
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
        wandb.finish()


#######################################
# 7. Register the callback and train
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

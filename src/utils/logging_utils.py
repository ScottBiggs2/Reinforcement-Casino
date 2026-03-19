
import os
import json
import csv
import torch
import wandb
from transformers import TrainerCallback

class FlexibleCheckpointCallback(TrainerCallback):
    """
    Callback that saves deltas on a flexible schedule and tracks statistics.
    
    Schedule: Every 5 steps for first 25 steps, then every 25 steps after.
    """
    
    def __init__(self, base_state, delta_log_dir, checkpoint_schedule, threshold, model_name, dataset_name, subset_size, learning_rate, batch_size, grad_accum, run_name=None, use_wandb=False, wandb_project=None):
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.checkpoint_schedule = set(checkpoint_schedule)  # Use set for O(1) lookup
        self.threshold = threshold
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.subset_size = subset_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.run_name = run_name or f"{model_name}_run"
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        os.makedirs(self.delta_log_dir, exist_ok=True)
        # If wandb is already active (e.g. via Trainer), mark as initialized
        self.wandb_initialized = (wandb.run is not None)

    def on_train_begin(self, args, state, control, **kwargs):
        # Only init if not already initialized (e.g. by Trainer)
        if self.use_wandb and not self.wandb_initialized and wandb.run is None:
            project_name = self.wandb_project if self.wandb_project else f"{self.model_name.replace('/', '_')}-dpo-subnetwork-emergence"
            wandb.init(
                project=project_name,
                name=self.run_name,
                config={
                    "model_name": self.model_name,
                    "dataset": self.dataset_name,
                    "subset_size": self.subset_size,
                    "learning_rate": self.learning_rate,
                    "batch_size_per_device": self.batch_size,
                    "grad_accum": self.grad_accum,
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
                if name in self.base_state:
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
        if self.use_wandb and self.wandb_initialized:
            all_l2 = [v["l2_from_init"] for v in layer_stats.values()]
            all_frac = [v["frac_big_from_init"] for v in layer_stats.values()]
            if all_l2:
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

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.wandb_initialized:
            wandb.finish()


class CSVLoggerCallback(TrainerCallback):
    """
    Logs per-step training metrics (loss, rewards) to a CSV file.
    """
    def __init__(self, output_dir, filename="training_log.csv"):
        self.output_dir = output_dir
        self.filepath = os.path.join(output_dir, filename)
        self.file = None
        self.writer = None
        os.makedirs(output_dir, exist_ok=True)
        
    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize file
        self.file = open(self.filepath, mode='w', newline='')
        self.writer = csv.writer(self.file)
        # We don't know exact header yet, will write on first log
        self.header_written = False
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.writer:
            # Flatten logs dict
            flat_logs = {"step": state.global_step}
            flat_logs.update(logs)
            
            # Write header if not written
            if not self.header_written:
                self.writer.writerow(flat_logs.keys())
                self.header_written = True
                
            # Write row
            self.writer.writerow(flat_logs.values())
            self.file.flush()
            
    def on_train_end(self, args, state, control, **kwargs):
        if self.file:
            self.file.close()
            print(f"✓ Training log saved to {self.filepath}")

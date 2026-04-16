
import os

# Avoid wandb patching sys.stdout for console capture when only callbacks import wandb;
# cluster stdouts (NFS-backed Slurm .out) can hit errno 116 on wrapped writes.
os.environ.setdefault("WANDB_CONSOLE", "off")

import json
import csv
import time
import torch
import wandb
import pandas as pd
from transformers import TrainerCallback
from typing import Any, Dict, List, Optional

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
        # Save run metadata for provenance/self-documentation
        metadata = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "subset_size": self.subset_size,
            "learning_rate": self.learning_rate,
            "batch_size_per_device": self.batch_size,
            "grad_accum": self.grad_accum,
            "checkpoint_schedule": sorted(list(self.checkpoint_schedule)),
            "run_name": self.run_name,
        }
        metadata_path = os.path.join(self.delta_log_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved run metadata to {metadata_path}")
        
        # Only init wandb if not already initialized (e.g. by Trainer)
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

        # Only iterate and clone parameters if we are on the checkpoint schedule.
        # This removes the massive per-step overhead of cloning 32GB to CPU.
        if step in self.checkpoint_schedule:
            full_deltas_to_save = {}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self.base_state:
                        current = param.detach().float().cpu()
                        diff = current - self.base_state[name]
                        full_deltas_to_save[name] = diff

            delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
            torch.save(full_deltas_to_save, delta_file)
            print(f"  ✓ Saved weight deltas at step {step}")

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


class BenchmarkRunLogSink:
    """
    Buffer benchmark rows in RAM, flush to CSV on demand.

    Avoids re-reading/rewriting the whole CSV on every Trainer log (slow and can trigger
    NFS errno 116 stale file handle under heavy scratch churn).
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._rows: List[Dict[str, Any]] = []

    def write_row(self, row: Dict[str, Any]) -> None:
        flat: Dict[str, Any] = {}
        for k, v in row.items():
            if v is None:
                flat[k] = ""
            elif isinstance(v, (bool, int, float, str)):
                flat[k] = v
            else:
                flat[k] = str(v)
        self._rows.append(flat)

    def flush(self) -> None:
        """Write all buffered rows to disk (atomic replace). Safe to call after each phase."""
        if not self._rows:
            return
        d = os.path.dirname(os.path.abspath(self.filepath)) or "."
        os.makedirs(d, exist_ok=True)
        df = pd.DataFrame(self._rows)
        base = os.path.basename(self.filepath)
        tmp = os.path.join(d, f".{base}.{os.getpid()}.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.filepath)

    def close(self) -> None:
        self.flush()


class BenchmarkThroughputCallback(TrainerCallback):
    """
    Per-log throughput (cumulative since phase start) plus phase metadata for plotting.
    """

    def __init__(
        self,
        sink: BenchmarkRunLogSink,
        phase: str,
        sparsity_target_pct: Optional[float],
        optimizer_label: str,
        print_every: int = 10,
    ):
        self.sink = sink
        self.phase = phase
        self.sparsity_target_pct = sparsity_target_pct
        self.optimizer_label = optimizer_label
        self._t0: Optional[float] = None
        self.print_every = int(print_every) if print_every and int(print_every) > 0 else 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._t0 is None:
            return
        wall = time.perf_counter() - self._t0
        step = int(state.global_step)
        world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        samples_per_step = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * world
        )
        sps = step / wall if wall > 0 else 0.0
        sms = (step * samples_per_step) / wall if wall > 0 else 0.0

        row: Dict[str, Any] = {
            "phase": self.phase,
            "sparsity_target_pct": (
                "" if self.sparsity_target_pct is None else float(self.sparsity_target_pct)
            ),
            "optimizer": self.optimizer_label,
            "step": step,
            "wall_time_s": round(wall, 6),
            "cumulative_steps_per_s": round(sps, 8),
            "cumulative_samples_per_s": round(sms, 6),
        }
        row.update(logs)
        self.sink.write_row(row)

        # Live timing printouts for Slurm .out monitoring.
        if self.print_every and step > 0 and (step % self.print_every == 0):
            sp = "dense" if self.sparsity_target_pct is None else f"{float(self.sparsity_target_pct):g}%"
            print(
                f"[throughput] phase={self.phase} sparsity={sp} step={step} "
                f"steps/s={sps:.4f} samples/s={sms:.2f} wall_s={wall:.1f}"
            )

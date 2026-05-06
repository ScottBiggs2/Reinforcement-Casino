
import os

# Avoid wandb patching sys.stdout for console capture when only callbacks import wandb;
# cluster stdouts (NFS-backed Slurm .out) can hit errno 116 on wrapped writes.
os.environ["WANDB_CONSOLE"] = "off"
os.environ.setdefault("WANDB_SILENT", "true")

import json
import csv
import time
import torch
import wandb
import pandas as pd
from transformers import TrainerCallback
from typing import Any, Dict, List, Optional

from src.utils.slurm_safe_log import slurm_safe_print


class OptimizerStepTimingCallback(TrainerCallback):
    """
    Pull optimizer.step() timing from a wrapped optimizer.

    We avoid patching Trainer internals; instead we wrap the optimizer's ``step`` method and
    read the recorded timings here on each log event.
    """

    def __init__(
        self,
        timed_optimizer: Any,
        *,
        sync_cuda: bool = True,
        prefix: str = "t_optimizer_step",
    ) -> None:
        self.opt = timed_optimizer
        self.sync_cuda = bool(sync_cuda)
        self.prefix = str(prefix)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        try:
            last_ms = float(getattr(self.opt, "last_step_ms", float("nan")))
        except Exception:
            last_ms = float("nan")
        try:
            mean_ms = float(getattr(self.opt, "mean_step_ms", float("nan")))
        except Exception:
            mean_ms = float("nan")
        try:
            n = int(getattr(self.opt, "step_count", 0) or 0)
        except Exception:
            n = 0

        if last_ms == last_ms:
            logs[f"{self.prefix}_ms"] = round(last_ms, 6)
        if mean_ms == mean_ms:
            logs[f"{self.prefix}_mean_ms"] = round(mean_ms, 6)
        if n:
            logs[f"{self.prefix}_count"] = n

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
        slurm_safe_print(f"  ✓ Saved run metadata to {metadata_path}")
        
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
            slurm_safe_print(f"  ✓ Saved weight deltas at step {step}")

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
            slurm_safe_print(f"✓ Training log saved to {self.filepath}")


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

    ``cumulative_steps_per_s`` / ``wall_time_s`` are measured from ``on_train_begin`` (HF trainer
    start only — excludes model load / injection when those ran before ``trainer.train()``).

    ``wall_time_since_first_step_end_s`` / ``cumulative_steps_per_s_excl_first`` start after the
    first optimizer step completes (reduces first-step compile skew in headline rates).

    For a **steady-interval** view, use ``wall_delta_s`` and ``inst_steps_per_s`` (delta since the
    previous logged row), when the trainer logs frequently enough that consecutive rows bracket
    completed optimizer steps.
    """

    def __init__(
        self,
        sink: BenchmarkRunLogSink,
        phase: str,
        sparsity_target_pct: Optional[float],
        optimizer_label: str,
        print_every: int = 10,
        extra_log_fields: Optional[Dict[str, Any]] = None,
    ):
        self.sink = sink
        self.phase = phase
        self.sparsity_target_pct = sparsity_target_pct
        self.optimizer_label = optimizer_label
        self._t0: Optional[float] = None
        self._t_after_step1: Optional[float] = None
        self.print_every = int(print_every) if print_every and int(print_every) > 0 else 0
        self._extra_log_fields = extra_log_fields or {}
        self._prev_wall: Optional[float] = None
        self._prev_step: Optional[int] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
        self._t_after_step1 = None
        self._prev_wall = None
        self._prev_step = None

    def on_step_end(self, args, state, control, **kwargs):
        if self._t_after_step1 is None and int(state.global_step) >= 1:
            self._t_after_step1 = time.perf_counter()

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

        ga = max(1, int(getattr(args, "gradient_accumulation_steps", 1) or 1))
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
            "trainer_grad_accum_steps": ga,
            "trainer_per_device_train_batch_size": int(args.per_device_train_batch_size),
        }
        if self._t_after_step1 is not None and step >= 1:
            w_ex = time.perf_counter() - self._t_after_step1
            row["wall_time_since_first_step_end_s"] = round(w_ex, 6)
            completed_after_first = max(0, step - 1)
            if w_ex > 1e-9 and completed_after_first > 0:
                row["cumulative_steps_per_s_excl_first"] = round(completed_after_first / w_ex, 8)
                row["cumulative_samples_per_s_excl_first"] = round(
                    (completed_after_first * samples_per_step) / w_ex, 6
                )
        # Interval rates (since previous log): closer to steady-state when logging every step.
        if self._prev_wall is not None and self._prev_step is not None:
            dw = wall - self._prev_wall
            dst = step - self._prev_step
            if dw > 1e-9 and dst > 0:
                row["wall_delta_s"] = round(dw, 6)
                row["inst_steps_per_s"] = round(dst / dw, 8)
                row["inst_samples_per_s"] = round((dst * samples_per_step) / dw, 6)
        self._prev_wall = wall
        self._prev_step = step

        row.update(self._extra_log_fields)
        row.update(logs)

        # Derived FLOPs/s (proxy): theory_bsr_backward_flops_proxy is per **optimizer step**
        # (b_tokens includes grad accum; see bsr_theory_metrics). t_backward_ms from detailed
        # timing is only the **last** micro-batch backward; scale by ga so divisor matches
        # one full step's backward work (assumes similar cost per micro-batch).
        try:
            flops = float(row.get("theory_bsr_backward_flops_proxy"))  # may be "" on dense phase
        except Exception:
            flops = None
        if flops is not None and flops > 0:
            try:
                bwd_ms = float(row.get("t_backward_ms"))
            except Exception:
                bwd_ms = None
            try:
                step_ms = float(row.get("t_step_total_ms"))
            except Exception:
                step_ms = None

            if bwd_ms is not None and bwd_ms > 0:
                bwd_step_ms = bwd_ms * float(ga)
                row["eff_bsr_backward_flops_per_s"] = round(flops / (bwd_step_ms / 1e3), 6)
                row["eff_bsr_backward_tflops"] = round((flops / (bwd_step_ms / 1e3)) / 1e12, 6)
            if step_ms is not None and step_ms > 0:
                row["eff_bsr_backward_flops_per_s_over_e2e_step"] = round(flops / (step_ms / 1e3), 6)
                row["eff_bsr_backward_tflops_over_e2e_step"] = round((flops / (step_ms / 1e3)) / 1e12, 6)

        self.sink.write_row(row)

        # Live timing printouts for Slurm .out monitoring.
        if self.print_every and step > 0 and (step % self.print_every == 0):
            sp = "dense" if self.sparsity_target_pct is None else f"{float(self.sparsity_target_pct):g}%"
            extra = ""
            ex = row.get("cumulative_steps_per_s_excl_first")
            if isinstance(ex, (int, float)) and ex == ex and ex > 0:
                extra = f" steps/s_excl_first={float(ex):.4f}"
            slurm_safe_print(
                f"[throughput] phase={self.phase} sparsity={sp} step={step} "
                f"steps/s={sps:.4f} samples/s={sms:.2f} wall_s={wall:.1f}{extra}"
            )

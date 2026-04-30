
#!/usr/bin/env python3
"""
Triton-Accelerated Sparse DPO Training - BSR BACKPROP FOCUSED

Refactored from sparse_DPO_v4.py.
Focus: Backprop Ablations (BSR Sparse MLP vs Dense MLP)

Key Features:
1. Injects BSR Sparse MLP Layers (SparseLinearLayer)
2. Uses Custom Sparse Autograd Function
3. Modular Architecture
"""

import os
import sys
import argparse
import types
from typing import Any, Dict, Optional

import torch

# Must override any inherited login-node exports before importing wandb (NFS Slurm logs + console_capture).
os.environ["WANDB_CONSOLE"] = "off"
os.environ.setdefault("WANDB_SILENT", "true")

import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOTrainer, DPOConfig

# Add project root to sys.path to resolve 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.mask_manager import SparseMaskManager
from src.utils.slurm_safe_log import slurm_safe_print
from src.utils.scratch_paths import default_hf_datasets_cache, default_rl_casino_outputs
from src.utils.data_utils import make_dpo_collator
from src.utils.dataset_registry import get_dataset_config, load_dpo_dataset as registry_load_dpo
from src.utils.logging_utils import (
    FlexibleCheckpointCallback,
    CSVLoggerCallback,
    BenchmarkThroughputCallback,
)
from src.optimizers.sparse_adamw import SparseAdamW
from src.mlps.bsr_sparse_mlp import replace_linear_modules, restore_linear_modules
from src.kernels.bsr_backward import bsr_recompile_diag_summary

def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized).strip("_")


def _dense_optimizer_torch_or_8bit(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    optim_preference: str,
):
    """
    Dense baseline optimizer. Pipeline `DPO_train.py` defaults to `adamw_8bit` (bitsandbytes).
    Set ``DPO_OPTIM=adamw_8bit`` to match; otherwise full ``torch.optim.AdamW``.
    """
    key = (optim_preference or "adamw").strip().lower().replace("-", "_")
    if key in ("adamw_8bit", "adamw8bit", "8bit"):
        try:
            from bitsandbytes.optim import AdamW8bit

            return AdamW8bit(
                model.parameters(),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
                weight_decay=weight_decay,
            )
        except ImportError:
            slurm_safe_print(
                "WARNING: DPO_OPTIM=adamw_8bit but bitsandbytes is not importable; "
                "using torch.optim.AdamW (install bitsandbytes for parity with DPO_train.py)."
            )
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
    )


def train(
    model_name,
    checkpoint_path,
    mask_path,
    n_steps,
    batch_size,
    learning_rate,
    subset_size,
    run_name,
    mlp_only,
    block_size_bsr,
    block_size_adam,
    optimizer_type,
    use_wandb,
    save_csv,
    grad_accum,
    max_grad_norm,
    adam_beta1,
    adam_beta2,
    adam_eps,
    dpo_beta,
    warmup_steps,
    warmup_ratio,
    weight_decay,
    max_length,
    max_prompt_length,
    disable_tf32,
    save_model,
    dataset_key,
    output_base_dir,
    dataset_cache_dir,
    dense_baseline: bool = False,
    no_delta_callback: bool = False,
    lr_scheduler_type: str = "linear",
    num_train_epochs: int = 1,
    gradient_checkpointing: bool = True,
    save_strategy: str = "no",
    report_to: str = "none",
    device_map="auto",
    train_dataset=None,
    tokenizer_obj=None,
    benchmark_log_sink=None,
    benchmark_phase: str = None,
    benchmark_sparsity_pct: float = None,
    benchmark_optimizer_label: str = None,
    benchmark_extra_log_fields: Optional[Dict[str, Any]] = None,
):
    # Determine paths
    if checkpoint_path is None or str(checkpoint_path).lower() == "none":
        checkpoint_path = model_name
    
    # Set dataset cache directory
    os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    
    # Resolve dataset via registry
    ds_config = get_dataset_config(dataset_key)
    dataset_name = ds_config["hf_id"]
    dataset_sanitized = ds_config["sanitized_name"]
    
    if run_name is None:
        # Construct a descriptive run name
        parts = ["sparse_dpo"]
        
        # Optimizer info
        if dense_baseline:
            parts.append("adamw")
        elif optimizer_type == "sparse_adamw":
            parts.append("bsr_adamw")
        else:
            parts.append(optimizer_type)

        if dense_baseline:
            parts.append("dense_backprop")
        else:
            parts.append("bsr_backprop")
        
        # Model info
        parts.append(sanitize_model_name(model_name))
        
        # Dataset info
        parts.append(dataset_sanitized)
        
        run_name = "_".join(parts)
    
    wandb_project = "huggingface"
    os.environ["WANDB_PROJECT"] = wandb_project
    run_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    slurm_safe_print(f"\n{'='*60}")
    slurm_safe_print(f"SPARSE DPO BSR TRAINING")
    slurm_safe_print(f"{'='*60}")
    slurm_safe_print(f"Run Directory: {run_dir}")
    slurm_safe_print(f"Dataset: {dataset_key} ({dataset_name})")
    slurm_safe_print(f"Block Size BSR: {block_size_bsr}")
    if dense_baseline:
        slurm_safe_print("Dense baseline: standard nn.Linear + dense AdamW (no BSR injection).")

    if tokenizer_obj is not None:
        tokenizer = tokenizer_obj
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if train_dataset is not None:
        dpo_dataset = train_dataset
    else:
        dpo_dataset = registry_load_dpo(dataset_key, subset_size=subset_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _device_map_for_pretrained(dm):
        if dm is None or dm == "auto":
            return "auto"
        if isinstance(dm, str) and dm.lower() in ("none", "null"):
            return None
        return dm

    load_dm = _device_map_for_pretrained(device_map)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=load_dm,
    )
    if load_dm is None and hasattr(model, "to") and device.type == "cuda":
        model.to(device)
    model.config.use_cache = False

    mask_manager = None
    mask_dict = {}
    if not dense_baseline:
        if not mask_path or not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Sparse training requires a valid mask file, got: {mask_path!r}")
        mask_manager = SparseMaskManager(mask_path, device=device)
        mask_dict = {
            n: mask_manager.get_mask(n)
            for n, _ in model.named_parameters()
            if ("mlp" in n.lower() or not mlp_only)
            and "weight" in n
            and mask_manager.has_mask(n)
        }

        slurm_safe_print(f"Injecting Sparse MLP BSR backward for {len(mask_dict)} layers...")
        use_tf32_kernel = not disable_tf32
        slurm_safe_print(f"BSR Kernel TF32 Precision Enabled: {use_tf32_kernel}")
        _quiet_inj = os.environ.get("RL_CASINO_BSR_QUIET_INJECTION", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        replace_linear_modules(
            model,
            mask_dict,
            block_size=block_size_bsr,
            use_tf32=use_tf32_kernel,
            verbose=not _quiet_inj,
        )
    else:
        slurm_safe_print("Skipping BSR layer injection (dense baseline).")

    eff_optimizer = "adamw" if dense_baseline else optimizer_type
    
    slurm_safe_print(f"Initializing {eff_optimizer}...")
    if eff_optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif eff_optimizer == "adamw":
        dense_pref = os.environ.get("DPO_OPTIM", os.environ.get("DPO_DENSE_OPTIM", "adamw"))
        optimizer = _dense_optimizer_torch_or_8bit(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
            optim_preference=dense_pref,
        )
    elif eff_optimizer == "sparse_adamw":
        if mask_manager is None:
            raise ValueError("sparse_adamw requires a mask (dense_baseline=False).")
        optimizer = SparseAdamW(
            list(model.named_parameters()),
            mask_manager,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            block_size=block_size_adam,
            mlp_only=mlp_only,
            max_grad_norm=max_grad_norm,
        )
    else:
        raise ValueError(f"Unknown optimizer: {eff_optimizer}")
        
    if disable_tf32:
        slurm_safe_print("Disabling TF32 for strict fp32 accumulation precision.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
    checkpoint_schedule = list(range(10, 50, 10)) + list(range(100, 250, 50))

    callbacks = []

    # ---------------------------------------------------------------------
    # Optional: segment timing for benchmarking (CUDA event based).
    #
    # Goal: separate optimizer-step time from the rest of the training step
    # (forward + backward + trainer overhead). This is critical because BSR
    # kernels only affect backward/optimizer, while end-to-end step time is
    # dominated by many other ops (attention, DPO mechanics, recompute).
    #
    # This callback augments the Trainer's `logs` dict so the existing
    # BenchmarkThroughputCallback writes timing columns into the same CSV.
    # ---------------------------------------------------------------------
    class _StepOptTimingCallback(TrainerCallback):
        def __init__(self):
            self._step_start_evt = None
            self._last_step_ms = None
            self._last_opt_ms = None

        def set_last_opt_ms(self, ms: float) -> None:
            self._last_opt_ms = float(ms)

        def on_step_begin(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                self._step_start_evt = torch.cuda.Event(enable_timing=True)
                self._step_start_evt.record()

        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available() and self._step_start_evt is not None:
                end_evt = torch.cuda.Event(enable_timing=True)
                end_evt.record()
                torch.cuda.synchronize()
                self._last_step_ms = float(self._step_start_evt.elapsed_time(end_evt))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if self._last_step_ms is not None:
                logs["t_step_total_ms"] = round(self._last_step_ms, 6)
            if self._last_opt_ms is not None:
                logs["t_optim_ms"] = round(self._last_opt_ms, 6)
                if self._last_step_ms is not None:
                    logs["t_nonoptim_ms"] = round(max(0.0, self._last_step_ms - self._last_opt_ms), 6)

    if not no_delta_callback:
        base_state = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                base_state[name] = param.detach().float().cpu().clone()

        callbacks.append(
            FlexibleCheckpointCallback(
                base_state=base_state,
                delta_log_dir=os.path.join(run_dir, "deltas"),
                checkpoint_schedule=checkpoint_schedule,
                threshold=1e-3,
                model_name=model_name,
                dataset_name=dataset_name,
                subset_size=subset_size,
                learning_rate=learning_rate,
                batch_size=batch_size,
                grad_accum=grad_accum,
                run_name=run_name,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
            )
        )

    if benchmark_log_sink is not None and benchmark_phase:
        label = benchmark_optimizer_label or eff_optimizer
        _te = os.environ.get("RL_CASINO_THROUGHPUT_PRINT_EVERY", "").strip()
        _print_every = int(_te) if _te else 25
        callbacks.append(
            BenchmarkThroughputCallback(
                benchmark_log_sink,
                phase=benchmark_phase,
                sparsity_target_pct=benchmark_sparsity_pct,
                optimizer_label=label,
                print_every=_print_every,
                extra_log_fields=benchmark_extra_log_fields,
            )
        )
        _timing_cb = _StepOptTimingCallback()
        callbacks.append(_timing_cb)
    elif save_csv:
        callbacks.append(CSVLoggerCallback(output_dir=run_dir))
        _timing_cb = _StepOptTimingCallback()
        callbacks.append(_timing_cb)
    else:
        _timing_cb = None

    warmup_kw = {}
    if warmup_steps and warmup_steps > 0:
        warmup_kw["warmup_steps"] = warmup_steps
    elif warmup_ratio and warmup_ratio > 0:
        warmup_kw["warmup_ratio"] = warmup_ratio

    rt = report_to
    if use_wandb:
        rt = "wandb"
    elif rt in (None, "", "none"):
        rt = "none"

    _log_steps_env = os.environ.get("RL_CASINO_LOGGING_STEPS", "").strip()
    logging_steps = int(_log_steps_env) if _log_steps_env else 1
    _disable_tqdm = os.environ.get("RL_CASINO_DISABLE_TQDM", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    dpo_config = DPOConfig(
        output_dir=os.path.join(run_dir, "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=n_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        disable_tqdm=_disable_tqdm,
        report_to=rt,
        run_name=run_name,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        beta=dpo_beta,
        weight_decay=weight_decay,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        **warmup_kw,
    )

    collate_fn = make_dpo_collator(tokenizer, max_prompt_length, max_length)

    # Wrap optimizer.step with CUDA-event timing so we can record optimizer-only time.
    #
    # IMPORTANT: assign a **bound method** (types.MethodType), not a bare function.
    # PyTorch 2.x LR schedulers patch optimizer.step via step_fn.__func__; plain functions
    # break LambdaLR construction (AttributeError: 'function' object has no attribute '__func__').
    # Kernel launches / Triton are unaffected—this only wraps the Optimizer.step call.
    if _timing_cb is not None and torch.cuda.is_available():
        _orig_step = optimizer.step  # bound method (AdamW / AdamW8bit / SparseAdamW)

        def _timed_step(self, closure=None):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                out = _orig_step(closure)
            finally:
                end.record()
                torch.cuda.synchronize()
                _timing_cb.set_last_opt_ms(float(start.elapsed_time(end)))
            return out

        optimizer.step = types.MethodType(_timed_step, optimizer)  # type: ignore[assignment]

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )
    
    trainer.train()

    # Optional: print BSR kernel compile/variant diagnostics for this phase.
    if os.environ.get("RL_CASINO_BSR_RECOMPILE_DIAG", "").strip().lower() in ("1", "true", "yes"):
        # Do not reset here by default; multi-phase benchmarks may want a global summary.
        slurm_safe_print(bsr_recompile_diag_summary(reset=False))

    # Final Saving and Cleanup
    if save_model:
        slurm_safe_print(f"\nTraining complete. Saving final model to {run_dir}/final_model...")
        # For BSR, we must restore the linear modules to dense (preserving weights)
        # so that the checkpoint is a standard HF-compatible model.
        restore_linear_modules(model)
        
        final_save_dir = os.path.join(run_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        
        # Save the model and tokenizer
        trainer.save_model(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)
        slurm_safe_print(f"✓ Full checkpoint saved to {final_save_dir}")
    else:
        slurm_safe_print("\nTraining complete. Skipping final model saving as requested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mask", type=str, default="masks/top_10.0pct_momentum_w25_step25.pt")
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "sparse_adamw"], default="sparse_adamw")
    parser.add_argument("--block_size_bsr", type=int, default=16)
    parser.add_argument("--block_size_adam", type=int, default=128)
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        default=False,
        help="Restrict sparse updates to MLP layers only (default: full model where masks exist)",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB and results directory")
    parser.add_argument("--dataset", type=str, default="light-r1",
                       help="Dataset key (light-r1, tulu3, math-step-dpo, codepref) or HuggingFace ID")
    parser.add_argument("--output_base_dir", type=str, default=default_rl_casino_outputs(), help="Base directory for outputs")
    parser.add_argument("--dataset_cache_dir", type=str, default=default_hf_datasets_cache(), help="Cache directory for HuggingFace datasets")
    
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser.add_argument("--save_model", type=str2bool, default=True, help="Save final model checkpoint (default: True)")
    
    # Stability Tuning Parameters
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon (increase to 1e-5 for stability)")
    
    # Advanced Noise Reduction
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO margin parameter (increase to 0.2-0.5 to bound updates)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup steps for LR scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup ratio (used if warmup_steps==0)")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=1024, help="Max token length for chosen/rejected in collator")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt tokens in collator")
    parser.add_argument("--dense_baseline", action="store_true", help="Dense AdamW without BSR injection")
    parser.add_argument("--no_delta_callback", action="store_true", help="Do not save weight delta checkpoints")
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map: auto, none (single GPU .to), or cuda:0",
    )
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_tf32", action="store_true", help="Disable TF32 for strict fp32 math (slow but precise)")
    
    args = parser.parse_args()

    dm = args.device_map
    if dm.lower() in ("none", "null"):
        dm = "none"

    gc = not args.no_gradient_checkpointing

    train(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        mask_path=args.mask,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_size=args.subset_size,
        run_name=args.run_name,
        mlp_only=args.mlp_only,
        block_size_bsr=args.block_size_bsr,
        block_size_adam=args.block_size_adam,
        optimizer_type=args.optimizer,
        use_wandb=args.use_wandb,
        save_csv=args.save_csv,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        dpo_beta=args.dpo_beta,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        disable_tf32=args.disable_tf32,
        save_model=args.save_model,
        dataset_key=args.dataset,
        output_base_dir=args.output_base_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        dense_baseline=args.dense_baseline,
        no_delta_callback=args.no_delta_callback,
        gradient_checkpointing=gc,
        device_map=dm,
    )

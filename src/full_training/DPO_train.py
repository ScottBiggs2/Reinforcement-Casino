import os
import json
import argparse
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any

from src.utils.scratch_paths import default_hf_datasets_cache, default_rl_casino_outputs
from src.utils.grpo_checkpoint_utils import (
    maybe_load_wandb_resume_env,
    resolve_resume_checkpoint,
    RunManifestCallback,
    WandbRunIdCallback,
)


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
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m-it",
        help="HuggingFace model name to load (default: google/gemma-3-270m-it)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="light-r1",
        help="Dataset key from registry (light-r1, tulu3, math-step-dpo, codepref) or HuggingFace ID",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=250,
        help="Number of training steps (default: 250; match pipeline NUM_STEPS_DPO)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="Train for N epochs (overrides --num_steps when set).",
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Limit dataset size (default: None = full)")
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=default_rl_casino_outputs(),
        help="Base directory for all outputs (checkpoints, deltas)",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=default_hf_datasets_cache(),
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Peak learning rate (default: 5e-7; match pipeline DPO_LEARNING_RATE).",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio for LR schedule.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max total sequence length.")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length.")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta.")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer name for Trainer (e.g. adamw_8bit).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing.")
    parser.add_argument(
        "--delta_log_interval",
        type=int,
        default=50,
        help="Save full weight deltas (vs init) every N steps for warm-start masks (default: 50).",
    )
    parser.add_argument(
        "--delta_log_end_step",
        type=int,
        default=None,
        help="Last training step (inclusive) to save deltas. Default: min(num_steps, max(interval, num_steps//10)) "
        "e.g. 10%% of run with interval 50 → steps 50..200 for 2k steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="HF Trainer checkpoint interval (save_strategy=steps). Omit or use a very large value to disable "
        "(delta-only mode, legacy pipeline behavior).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Keep only the newest K HF checkpoints on disk when --save_steps is set (rolling). Default: 3.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint-* directory, or 'auto' for latest under output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir

    model_name = args.model_name
    model_name_sanitized = sanitize_model_name(model_name)

    from src.utils.dataset_registry import get_dataset_config

    dataset_key = args.dataset
    dataset_config = get_dataset_config(dataset_key)
    dataset_name = dataset_config["hf_id"]
    dataset_sanitized = dataset_config["sanitized_name"]

    base_dir = args.output_base_dir
    sub_dir = f"{model_name_sanitized}_{dataset_sanitized}"
    output_dir = os.path.join(base_dir, "checkpoints", sub_dir)
    delta_log_dir = os.path.join(base_dir, "deltas", sub_dir)

    os.makedirs(output_dir, exist_ok=True)

    resume_ckpt = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    if args.use_wandb:
        maybe_load_wandb_resume_env(base_dir, resume_ckpt)

    num_steps = args.num_steps
    subset_size = args.subset_size
    num_epochs = args.num_train_epochs

    interval = args.delta_log_interval
    end = args.delta_log_end_step
    if end is None:
        end = min(num_steps, max(interval, num_steps // 10))
    else:
        end = min(num_steps, end)
    checkpoint_schedule = list(range(interval, end + 1, interval))
    if not checkpoint_schedule and num_steps > 0:
        checkpoint_schedule = [num_steps]

    wandb_project = "huggingface"
    os.environ["WANDB_PROJECT"] = wandb_project
    wandb_run_name = args.run_name if args.run_name else f"{model_name_sanitized}_{dataset_sanitized}_dpo_{num_steps}steps"

    print(f"Delta (warm-mask) schedule: {checkpoint_schedule}")
    if resume_ckpt:
        print(f"Resume: {resume_ckpt!r} — skipping weight-delta callback (base_state would not match a cold start).")

    from src.utils.dataset_registry import load_dpo_dataset as registry_load_dpo

    raw_ds = registry_load_dpo(dataset_key, subset_size=subset_size)
    train_dataset = raw_ds
    eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def dpo_collator_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
                "prompt_input_ids": p_ids,
                "prompt_attention_mask": p_mask,
                "chosen_input_ids": c_ids,
                "chosen_attention_mask": c_mask,
                "rejected_input_ids": r_ids,
                "rejected_attention_mask": r_mask,
            }

        prompts = [ex.get("prompt", "") for ex in examples]
        chosens = [ex.get("chosen", "") for ex in examples]
        rejects = [ex.get("rejected", "") for ex in examples]

        _mpl = args.max_prompt_length
        _ml = args.max_length
        enc_prompt = [tokenizer(p, truncation=True, max_length=_mpl, return_tensors="pt") for p in prompts]
        enc_chosen = [tokenizer(c, truncation=True, max_length=_ml, return_tensors="pt") for c in chosens]
        enc_reject = [tokenizer(r, truncation=True, max_length=_ml, return_tensors="pt") for r in rejects]

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.config.use_cache = False

    _grad_ckpt = args.gradient_checkpointing
    if args.no_gradient_checkpointing:
        _grad_ckpt = False

    save_steps_arg = args.save_steps
    save_total_limit_arg = args.save_total_limit if args.save_total_limit is not None else 3
    use_hf_rolling = save_steps_arg is not None and save_steps_arg > 0 and save_steps_arg < 10**9

    if use_hf_rolling:
        save_strategy = "steps"
        hf_save_steps = save_steps_arg
        hf_save_total_limit = save_total_limit_arg
    else:
        save_strategy = "no"
        hf_save_steps = 500
        hf_save_total_limit = None

    cfg = DPOConfig(
        output_dir=output_dir,
        run_name=wandb_run_name,
        report_to=["wandb"] if args.use_wandb else [],
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        max_steps=-1 if num_epochs is not None else num_steps,
        num_train_epochs=num_epochs if num_epochs is not None else 1,
        bf16=True,
        fp16=False,
        optim=args.optim,
        gradient_checkpointing=_grad_ckpt,
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=hf_save_steps,
        save_total_limit=hf_save_total_limit,
        remove_unused_columns=False,
        beta=args.dpo_beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    print(
        f"DPO training: max_steps={cfg.max_steps}, num_train_epochs={cfg.num_train_epochs}, "
        f"peak_lr={args.learning_rate}, warmup_ratio={args.warmup_ratio}, lr_scheduler=linear"
    )
    print(f"HF rolling checkpoints: {use_hf_rolling} (save_steps={save_steps_arg}, limit={hf_save_total_limit})")

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=dpo_collator_fn,
    )

    manifest = {
        "model_name": model_name,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "subset_size": subset_size,
        "num_steps": num_steps,
        "learning_rate": args.learning_rate,
        "dpo_beta": args.dpo_beta,
        "optim": args.optim,
        "output_dir": output_dir,
        "resume_from_checkpoint": resume_ckpt,
        "hf_rolling_save_steps": save_steps_arg,
        "hf_save_total_limit": hf_save_total_limit if use_hf_rolling else None,
    }
    trainer.add_callback(RunManifestCallback(base_dir, manifest))

    if args.use_wandb:
        trainer.add_callback(WandbRunIdCallback(base_dir))

    class FlexibleCheckpointCallback(TrainerCallback):
        """Saves weight deltas vs θ(0) for warm-start masks (omit when resuming from HF checkpoint)."""

        def __init__(
            self,
            base_state: Dict[str, torch.Tensor],
            delta_log_dir: str,
            checkpoint_schedule: List[int],
            wandb_project_name: str,
        ):
            self.base_state = base_state
            self.delta_log_dir = delta_log_dir
            self.checkpoint_schedule = set(checkpoint_schedule)
            self.wandb_project_name = wandb_project_name
            os.makedirs(self.delta_log_dir, exist_ok=True)
            self.wandb_initialized = False

        def on_train_begin(self, train_args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            if not self.wandb_initialized and "wandb" in train_args.report_to:
                wandb.init(
                    project=self.wandb_project_name,
                    name=train_args.run_name,
                    config={
                        "model_name": model_name,
                        "dataset": dataset_name,
                        "subset_size": subset_size,
                        "learning_rate": train_args.learning_rate,
                        "batch_size_per_device": train_args.per_device_train_batch_size,
                        "grad_accum": train_args.gradient_accumulation_steps,
                        "checkpoint_schedule": sorted(list(self.checkpoint_schedule)),
                    },
                )
                self.wandb_initialized = True

        def on_step_end(self, train_args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            train_model = kwargs["model"]
            step = state.global_step
            if step in self.checkpoint_schedule:
                full_deltas_to_save = {}
                with torch.no_grad():
                    for name, param in train_model.named_parameters():
                        current = param.detach().float().cpu()
                        diff = current - self.base_state[name]
                        full_deltas_to_save[name] = diff
                delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
                torch.save(full_deltas_to_save, delta_file)
                print(f"  ✓ Saved weight deltas at step {step}")
            return control

        def on_train_end(self, train_args, state, control, **kwargs):
            if state.is_world_process_zero and self.wandb_initialized:
                wandb.finish()

    if not resume_ckpt:
        base_state: Dict[str, torch.Tensor] = {}
        if trainer.is_world_process_zero():
            with torch.no_grad():
                for name, param in trainer.model.named_parameters():
                    base_state[name] = param.detach().float().cpu().clone()
            os.makedirs(delta_log_dir, exist_ok=True)
            torch.save(base_state, os.path.join(delta_log_dir, "base_state.pt"))
        trainer.add_callback(
            FlexibleCheckpointCallback(
                base_state=base_state,
                delta_log_dir=delta_log_dir,
                checkpoint_schedule=checkpoint_schedule,
                wandb_project_name=wandb_project,
            )
        )

    print(f"\n{'=' * 60}")
    print("Starting DPO training")
    print(f"{'=' * 60}")
    if not resume_ckpt:
        print(f"Delta checkpoints (warm masks) at steps: {checkpoint_schedule}")
    print(f"Total steps target: {num_steps}")
    print(f"{'=' * 60}\n")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")
    print(f"Deltas dir: {delta_log_dir}")
    print(f"HF checkpoints: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

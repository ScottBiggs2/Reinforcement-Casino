# GRPO import failures (TRL + vLLM)

## Symptom

```
RuntimeError: Failed to import trl.trainer.grpo_trainer ...
Tried to instantiate class '_core_C.ScalarType' ...
```

or vLLM errors such as `undefined symbol` in `vllm/_core_C.abi3.so`.

## Cause

TRL 0.24’s `GRPOTrainer` loads optional vLLM integration when the `vllm` package is importable. A vLLM wheel built for a **different PyTorch** than the one in your conda env can fail during import, **before** any `use_vllm` flag is read. That breaks all GRPO entrypoints.

This is an environment issue, not a branch-specific code difference versus `RL-irene` (same `trl` / `torch` pins in `requirements.txt`).

## Fix (pick one)

1. **Default in this repo:** `src/utils/trl_vllm_import_guard.py` runs before `from trl import GRPOTrainer` and skips vLLM unless you set `TRL_SKIP_VLLM_IMPORT=0`. Training uses Hugging Face generate for rollouts (`use_vllm=false` in config).

2. **Remove broken vLLM:** `pip uninstall vllm` in the training env (use a separate env for eval jobs that need vLLM).

3. **Align versions:** Install a vLLM build that matches your PyTorch; TRL’s import utils warn that only certain vLLM versions are tested (see [TRL vLLM integration](https://huggingface.co/docs/trl/vllm_integration)).

## Verify job

`scripts/verify_grpo_training.sh` sets `TRL_SKIP_VLLM_IMPORT=1` and uses `src.utils.model_slug` for run slugs so helper `python -c` snippets never import TRL.

---

## `ValueError: Your setup doesn't support bf16/gpu`

### Symptom

Hugging Face `TrainingArguments` / `GRPOConfig` raises this during `__post_init__` when `bf16=True` but the **GPU + PyTorch build** does not pass Transformers’ bf16 training check (common on **V100** and some partitions).

### Fix

- **`--precision auto`** (default): uses bf16 when `torch.cuda.is_bf16_supported()` is true, otherwise **fp16** for the Trainer and `torch.float16` for model load.
- **`--precision fp16`**: force fp16 if auto still mis-detects.
- **`--precision bf16`**: force bf16 on GPUs where it is supported (e.g. A100, H200).

Implemented in [`src/utils/training_precision.py`](../src/utils/training_precision.py) for [`GRPO_train.py`](../src/full_training/GRPO_train.py) and [`sparse_grpo_bsr.py`](../src/full_training/sparse_grpo_bsr.py).

---

## W&B: `train/rewards/format_reasoning_reward` stuck at zero

### Cause

Rewards in [`src/utils/grpo_rewards.py`](../src/utils/grpo_rewards.py) used to assume `<redacted_thinking>...</redacted_thinking>`. Instruct models often **never** emit those tags, so the reasoning term was always zero.

### Fix

- Use **`--grpo_reward_profile llama_cot`** (default) or `export GRPO_REWARD_PROFILE=llama_cot` so parsing uses **delimiter-based** splitting as well as tags when present.
- Or use **`openr1_tags`** and rely on the appended prompt instruction (see `OPENR1_TAG_PROMPT_SUFFIX` in `grpo_rewards.py`) so the model is nudged toward redacted blocks.

---

## Eval scores look worse than training suggested

Benchmark tasks may use a **shorter** max generation than your training **`max_completion_length`**. Align lm-eval / task `max_gen_toks` with training when comparing GRPO checkpoints to baselines, or note both caps in reports.

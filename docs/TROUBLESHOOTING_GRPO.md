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

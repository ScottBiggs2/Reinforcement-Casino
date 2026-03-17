"""
MMLU (Massive Multitask Language Understanding) evaluation harness.
"""

import os
import sys
from typing import Dict, Any, Optional
import torch

# IMPORTANT WORKAROUND for Python 3.11 Dataclass / Pydantic Poison Bug
# We MUST import datasets BEFORE vLLM or lm_eval is loaded
try:
    import datasets
except ImportError:
    pass

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Benchmark-specific imports are moved inside functions to prevent dependency issues 
# from crashing the entire suite.

try:
    from lm_eval import simple_evaluate
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _has_chat_template(model_path: str) -> bool:
    """Check if a model has a chat template by inspecting its tokenizer config."""
    if not TRANSFORMERS_AVAILABLE:
        return False
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer.chat_template is not None
    except Exception:
        return False


def evaluate_mmlu(
    model_path: str,
    model: str = "hf",
    num_fewshot: int = 5,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: Any = "auto",
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on the MMLU benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        model: Model backend ("hf" or "vllm")
        num_fewshot: Number of few-shot examples (default: 5)
        limit: Limit number of examples per task (None = all)
        device: Device to run on (auto-detect if None)
        dtype: Model dtype (auto-detect if None)
        batch_size: Batch size for evaluation (can be "auto")
        trust_remote_code: Whether to trust remote code
        apply_chat_template: Whether to apply the model's chat template (auto for instruct/chat if None)
        
    Returns:
        Dictionary with evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-evaluation-harness is required for MMLU evaluation. "
            "Install with: pip install lm-eval"
        )
    
    if verbose:
        print("=" * 60)
        print("MMLU EVALUATION")
        print("=" * 60)
    
    # Auto-detect device if not specified
    if device is None:
        if model == "vllm":
            device = "cuda"
        elif torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Auto-detect dtype if not specified
    if dtype is None:
        if model == "vllm":
            dtype_str = "float16"
        elif device == "cuda" and torch.cuda.is_available():
            dtype_str = "float16"
        elif device == "mps" and torch.backends.mps.is_available():
            dtype_str = "float16"
        else:
            dtype_str = "float32"
    else:
        dtype_str = str(dtype).replace("torch.", "")
    
    # Auto-apply chat templates for instruct/chat models if not explicitly set
    # NOTE: This codebase uses instruct models, so we should be confident about applying templates
    if apply_chat_template is None:
        lower_path = model_path.lower()
        # First check path-based detection for HuggingFace models
        path_has_instruct = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])
        
        # For local paths, check the model config to see if it has a chat template
        # (local models saved from instruct training will have chat templates in config)
        if os.path.exists(model_path):
            if path_has_instruct:
                # Path indicates instruct model, apply template
                apply_chat_template = True
                if verbose:
                    print(f"✓ Detected instruct model from path, enabling chat template")
            else:
                # Check config to see if model has chat template (for local saved models)
                if _has_chat_template(model_path):
                    apply_chat_template = True
                    if verbose:
                        print(f"✓ Model has chat template in config (instruct model), enabling chat template")
                else:
                    apply_chat_template = False
                    if verbose:
                        print(f"⚠ Model does not have chat template in config, disabling chat template")
        else:
            # HuggingFace model - use path-based detection
            apply_chat_template = path_has_instruct
            if verbose:
                if apply_chat_template:
                    print(f"✓ Detected instruct model from HuggingFace path, enabling chat template")
                else:
                    print(f"⚠ HuggingFace model path doesn't indicate instruct model, disabling chat template")
        
        if verbose:
            print(f"Final decision: apply_chat_template = {apply_chat_template}")
    elif verbose:
        print(f"Chat template explicitly set to: {apply_chat_template}")

    # Convert to absolute path if it's a local path (for lm-eval compatibility)
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    
    # Lazy import for stability
    try:
        from lm_eval import simple_evaluate
    except ImportError:
        raise ImportError("lm-evaluation-harness is required for MMLU evaluation.")
        
    # Build model_args string for lm-eval
    # lm-eval's from_pretrained will automatically detect and use .safetensors files
    base_model_args_parts = [f"pretrained={model_path}", f"dtype={dtype_str}"]
    if trust_remote_code:
        base_model_args_parts.append("trust_remote_code=True")
    
    # vLLM robustness flags
    if model == "vllm":
        # Explicit max_model_len to avoid auto-derivation bugs
        base_model_args_parts.append("max_model_len=4096")
        # Disable chunked prefill which can cause NoneType errors in 0.6.3
        base_model_args_parts.append("enable_chunked_prefill=False")
        # Explicitly set max_num_batched_tokens to avoid NoneType comparison in scheduler
        base_model_args_parts.append("max_num_batched_tokens=4096")
        # Explicitly set max_num_seqs because lm-eval defaults it to None, crashing 0.6.3
        base_model_args_parts.append("max_num_seqs=256")
        
    base_model_args_str = ",".join(base_model_args_parts)
    
    # Try different configurations for chat template
    configs_to_try = []
    if apply_chat_template:
        configs_to_try.append({"apply_chat_template": True, "fewshot_as_multiturn": True})
        configs_to_try.append({"apply_chat_template": True})
    configs_to_try.append({})  # Base config without chat template
    
    # Run evaluation using lm-evaluation-harness
    # Note: simple_evaluate will load the model internally
    if verbose:
        print(f"\nRunning MMLU evaluation with {num_fewshot}-shot learning...")
        print(f"Model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype_str}")
        print(f"Chat template: {apply_chat_template}")
        print(f"Generation params: temperature=0.0, max_gen_toks=256")
        if limit:
            print(f"Limiting to {limit} examples per task")
    else:
        print("MMLU: Running...", end=" ", flush=True)
    
    results = None
    last_error = None
    
    import logging
    import warnings
    lm_eval_logger = logging.getLogger("lm_eval")
    old_level = lm_eval_logger.level
    if not verbose:
        lm_eval_logger.setLevel(logging.WARNING)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*chat template.*")
        
        for config in configs_to_try:
            try:
                eval_kwargs = {
                    "model": model,
                    "model_args": base_model_args_str,
                    "tasks": "mmlu",
                    "num_fewshot": num_fewshot,
                    "limit": limit,
                    "batch_size": batch_size,
                }
                # Only pass device for non-vllm models (vllm handles devices internally)
                if model != "vllm":
                    eval_kwargs["device"] = device
                    
                eval_kwargs.update(config)
                
                # Filter out None values to prevent library-level crashes on comparisons
                filtered_eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
                
                results = simple_evaluate(**filtered_eval_kwargs)
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                if "apply_chat_template" in str(e) or "fewshot_as_multiturn" in str(e):
                    if verbose:
                        print(f"Chat template config not supported: {config}; retrying without it.")
                    last_error = e
                    continue
                raise
    
    if not verbose:
        lm_eval_logger.setLevel(old_level)

    if results is None:
        raise last_error or RuntimeError("Failed to run MMLU evaluation.")
    
    # Extract and print key metrics
    if "results" in results:
        mmlu_results = results["results"]
        if "mmlu" in mmlu_results:
            mmlu_score = mmlu_results["mmlu"].get("acc,none", 0)
            if verbose:
                print("\n" + "=" * 60)
                print("MMLU RESULTS")
                print("=" * 60)
                print(f"\nMMLU Accuracy: {mmlu_score:.4f}")
            else:
                print(f"Accuracy: {mmlu_score:.4f}")
        else:
            # MMLU has multiple subtasks
            subtask_scores = {}
            for key, value in mmlu_results.items():
                if "mmlu" in key.lower():
                    if "acc" in value:
                        subtask_scores[key] = value["acc"]
            if subtask_scores:
                avg_score = sum(subtask_scores.values()) / len(subtask_scores)
                if verbose:
                    print("\n" + "=" * 60)
                    print("MMLU RESULTS")
                    print("=" * 60)
                    print(f"\nMMLU Average Accuracy: {avg_score:.4f}")
                    print("\nPer-subject scores:")
                    for subject, score in sorted(subtask_scores.items()):
                        print(f"  {subject}: {score:.4f}")
                else:
                    print(f"Accuracy: {avg_score:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HuggingFace ID or local path)")
    parser.add_argument("--num_fewshot", type=int, default=5,
                        help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (auto-detect if not specified)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code in model config")
    
    args = parser.parse_args()
    
    results = evaluate_mmlu(
        model_path=args.model_path,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

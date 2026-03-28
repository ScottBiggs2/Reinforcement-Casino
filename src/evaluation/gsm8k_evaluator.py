# GSM8K benchmark evaluation harness.
import os
import sys
from typing import Dict, Any, Optional
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.model_args_utils import build_lm_eval_model_args_parts

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


def evaluate_gsm8k(
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
    Evaluate a model on the GSM8K benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        model: Model backend ("hf" or "vllm")
        num_fewshot: Number of few-shot examples (default: 5)
        limit: Limit number of examples (None = all)
        device: Device to run on (auto-detect if None)
        dtype: Model dtype (auto-detect if None)
        batch_size: Batch size for evaluation (can be "auto")
        trust_remote_code: Whether to trust remote code
        apply_chat_template: Whether to apply the model's chat template
        
    Returns:
        Dictionary with evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-evaluation-harness is required for GSM8K evaluation. "
            "Install with: pip install lm-eval"
        )
    
    if verbose:
        print("=" * 60)
        print("GSM8K EVALUATION")
        print("-" * 60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print("-" * 60)
        print("Environment Variables:")
        for k, v in os.environ.items():
            if k.startswith(("VLLM_", "HF_", "CUDA_", "PYTHON")):
                print(f"  {k}: {v}")
        print("=" * 60)
    
    # Auto-detect device if not specified
    if device is None:
        if model == "vllm":
            device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect dtype if not specified
    if dtype is None:
        if model == "vllm":
            dtype_str = "float16"
        else:
            dtype_str = "float16" if device == "cuda" and torch.cuda.is_available() else "float32"
    else:
        dtype_str = str(dtype).replace("torch.", "")
    
    # Auto-apply chat templates for instruct/chat models
    if apply_chat_template is None:
        lower_path = model_path.lower()
        path_has_instruct = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])
        
        if os.path.exists(model_path):
            apply_chat_template = path_has_instruct or _has_chat_template(model_path)
        else:
            apply_chat_template = path_has_instruct
    
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    
    # Ensure model_path doesn't already have 'pretrained=' prefix
    clean_model_path = model_path
    if model_path.startswith("pretrained="):
        clean_model_path = model_path.replace("pretrained=", "", 1)
        
    base_model_args_parts = build_lm_eval_model_args_parts(
        model_path=clean_model_path,
        dtype_str=dtype_str,
        backend=model,
        trust_remote_code=trust_remote_code,
    )
    
    # vLLM robustness flags
    if model == "vllm":
        # Explicit max_model_len to avoid auto-derivation bugs
        base_model_args_parts.append("max_model_len=4096")
        # Disable chunked prefill which can cause NoneType errors in 0.6.3
        base_model_args_parts.append("enable_chunked_prefill=False")
        # Explicitly set max_num_batched_tokens to avoid NoneType comparison in scheduler
        base_model_args_parts.append("max_num_batched_tokens=4096")
        # Explicitly set max_num_seqs because lm-eval defaults it to None, crashing 0.6.3
        base_model_args_parts.append("max_num_seqs=64")
        base_model_args_parts.append("gpu_memory_utilization=0.7")
        
    base_model_args_str = ",".join(base_model_args_parts)
    
    if verbose:
        print(f"\nRunning GSM8K evaluation with {num_fewshot}-shot learning...")
        print(f"Model: {model_path}")
        print(f"Chat template: {apply_chat_template}")
        if limit:
            print(f"Limiting to {limit} examples")
    
    eval_kwargs = {
        "model": model,
        "model_args": base_model_args_str,
        "tasks": "gsm8k",
        "num_fewshot": num_fewshot,
        "limit": limit,
        "batch_size": batch_size,
        "gen_kwargs": {
            "temperature": 0.0,
            "max_gen_toks": 256,
        }
    }
    # Only pass device for non-vllm models (vllm handles devices internally)
    if model != "vllm":
        eval_kwargs["device"] = device
    
    # Try different configurations for chat template
    configs_to_try = []
    if apply_chat_template:
        configs_to_try.append({"apply_chat_template": True, "fewshot_as_multiturn": True})
        configs_to_try.append({"apply_chat_template": True})
    configs_to_try.append({})  # Base config without chat template
    
    # Lazy import for stability
    from lm_eval import simple_evaluate
    
    results = None
    last_error = None
    
    for config in configs_to_try:
        try:
            current_eval_kwargs = eval_kwargs.copy()
            current_eval_kwargs.update(config)
            
            # Filter out None values to prevent library-level crashes on comparisons
            filtered_eval_kwargs = {k: v for k, v in current_eval_kwargs.items() if v is not None}
            
            results = simple_evaluate(**filtered_eval_kwargs)
            break
        except Exception as e:
            import traceback
            error_msg = str(e)
            
            # Check for known chat template or fewshot issues
            chat_template_errors = [
                "apply_chat_template",
                "fewshot_as_multiturn",
                "Answer is not a string",
                "AssertionError"
            ]
            
            if any(err in error_msg for err in chat_template_errors) or isinstance(e, AssertionError):
                if verbose:
                    traceback.print_exc()
                    print(f"Chat template config or mult-turn fewshot not supported: {config}; retrying with simpler config.")
                last_error = e
                continue
            
            traceback.print_exc()
            raise e
            
    if results is None:
        raise last_error or RuntimeError("Failed to run GSM8K evaluation.")
    
    # Extract score
    gsm8k_score = None
    if "results" in results and "gsm8k" in results["results"]:
        task_result = results["results"]["gsm8k"]
        for acc_key in ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "exact_match", "acc,none", "acc"]:
            if acc_key in task_result:
                gsm8k_score = task_result[acc_key]
                break
                
    if gsm8k_score is not None:
        if verbose:
            print("\n" + "=" * 60)
            print("GSM8K RESULTS")
            print("=" * 60)
            print(f"\nGSM8K Accuracy: {gsm8k_score:.4f}")
        else:
            print(f"GSM8K Accuracy: {gsm8k_score:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--trust_remote_code", action="store_true")
    
    args = parser.parse_args()
    evaluate_gsm8k(
        model_path=args.model_path,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
    )

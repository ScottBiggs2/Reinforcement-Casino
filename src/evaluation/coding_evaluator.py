"""
Coding benchmarks evaluation harness (MBPP, HumanEval).
"""

import os
import sys
from typing import Dict, Any, Optional
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from lm_eval import simple_evaluate
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    print("Warning: lm-evaluation-harness not installed.")

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


def evaluate_coding(
    model_path: str,
    model: str = "hf",
    tasks: str = "humaneval,mbpp",
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: Any = 1, # Coding tasks usually batch_size=1
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on coding benchmarks (HumanEval, MBPP).
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        model: Model backend ("hf" or "vllm")
        tasks: Comma-separated list of tasks (default: "humaneval,mbpp")
        num_fewshot: Number of few-shot examples (default: 0 for coding)
        limit: Limit number of examples (None = all)
        device: Device to run on
        dtype: Model dtype
        batch_size: Batch size (can be "auto")
        trust_remote_code: Whether to trust remote code
        apply_chat_template: Whether to apply the model's chat template
        
    Returns:
        Dictionary with evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError("lm-evaluation-harness is required for coding evaluation.")
    
    if verbose:
        print("=" * 60)
        print(f"CODING EVALUATION: {tasks}")
        print("=" * 60)
    
    # Auto-detect device
    if device is None:
        if model == "vllm":
            device = "cuda" # vLLM always uses cuda
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect dtype
    if dtype is None:
        if model == "vllm":
            dtype_str = "float16" # Common for vLLM
        else:
            dtype_str = "float16" if device == "cuda" and torch.cuda.is_available() else "float32"
    else:
        dtype_str = str(dtype).replace("torch.", "")
    
    # Auto-apply chat templates
    if apply_chat_template is None:
        lower_path = model_path.lower()
        path_has_instruct = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])
        if os.path.exists(model_path):
            apply_chat_template = path_has_instruct or _has_chat_template(model_path)
        else:
            apply_chat_template = path_has_instruct
    
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    
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
    
    if verbose:
        print(f"\nRunning coding evaluation for tasks: {tasks}")
        print(f"Model: {model_path}")
        print(f"Chat template: {apply_chat_template}")
    
    # WARNING: Native HumanEval/MBPP in lm-eval uses execution-based evaluation
    # This requires 'allow_code_execution=True' in recent lm-eval versions
    eval_kwargs = {
        "model": model,
        "model_args": base_model_args_str,
        "tasks": tasks.split(","),
        "num_fewshot": num_fewshot,
        "limit": limit,
        "batch_size": batch_size,
        "gen_kwargs": {
            "temperature": 0.0,
            "max_gen_toks": 512,
            "do_sample": False,
        }
    }
    
    # Only pass device for non-vllm models (vllm handles devices internally)
    if model != "vllm":
        eval_kwargs["device"] = device
    
    if apply_chat_template:
        eval_kwargs["apply_chat_template"] = True
        
    # Attempt to run with code execution allowed if the version supports it
    # We use a more robust way to check for the argument to avoid the crash in the worker
    import inspect
    sig = inspect.signature(simple_evaluate)
    if "allow_code_execution" in sig.parameters:
        eval_kwargs["allow_code_execution"] = True

    try:
        # Filter out None values to prevent library-level crashes on comparisons
        filtered_eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
        results = simple_evaluate(**filtered_eval_kwargs)
    except Exception as e:
        if "marked as unsafe" in str(e) or "allow_code_execution" in str(e) or "confirm_run_unsafe_code" in str(e):
            if verbose:
                print("Task marked as unsafe, retrying with confirm_run_unsafe_code=True...")
            # Set the environment variable just in case
            import os
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            
            filtered_eval_kwargs["confirm_run_unsafe_code"] = True
            
            try:
                # Retry with confirm_run_unsafe_code
                results = simple_evaluate(**filtered_eval_kwargs)
            except TypeError as te:
                if "unexpected keyword argument" in str(te):
                    # If kwargs is rejected, try with allow_code_execution=True
                    del filtered_eval_kwargs["confirm_run_unsafe_code"]
                    filtered_eval_kwargs["allow_code_execution"] = True
                    results = simple_evaluate(**filtered_eval_kwargs)
                else:
                    raise
        else:
            import traceback
            if verbose:
                print(f"Error in simple_evaluate: {e}")
            traceback.print_exc()
            raise e
        
    if verbose and "results" in results:
        print("\n" + "=" * 60)
        print("CODING RESULTS (pass@1)")
        print("=" * 60)
        for task, task_results in results["results"].items():
            pass_at_1 = task_results.get("pass@1", task_results.get("acc", "N/A"))
            print(f"{task}: {pass_at_1}")
            
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on Coding benchmarks")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="humaneval,mbpp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    
    args = parser.parse_args()
    evaluate_coding(
        model_path=args.model_path,
        tasks=args.tasks,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )

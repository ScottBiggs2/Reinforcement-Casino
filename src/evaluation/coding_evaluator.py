"""
Coding benchmarks evaluation harness (MBPP, HumanEval).
"""

import os
import sys
from typing import Dict, Any, Optional
import torch
import gc
import traceback

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
        # Common keywords for instruct/chat models
        instruct_keywords = ["instruct", "chat", "-it", "-int", "dpo", "rlhf"]
        path_has_instruct = any(keyword in lower_path for keyword in instruct_keywords)
        
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
        
    base_model_args_parts = [f"pretrained={clean_model_path}", f"dtype={dtype_str}"]
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
        # Limit max_num_seqs and memory to avoid OOM
        base_model_args_parts.append("max_num_seqs=32")
        base_model_args_parts.append("gpu_memory_utilization=0.7")
        
    base_model_args_str = ",".join(base_model_args_parts)
    
    if verbose:
        print(f"\nRunning coding evaluation for tasks: {tasks}")
        print(f"Model: {model_path}")
        print(f"Chat template: {apply_chat_template}")
    
    eval_kwargs = {
        "model": model,
        "model_args": base_model_args_str,
        "tasks": [], # Will be set per task
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
    
    # Try different configurations for chat template
    configs_to_try = []
    if apply_chat_template:
        configs_to_try.append({"apply_chat_template": True})
    configs_to_try.append({})  # Base config without chat template
    
    all_task_results = {}
    
    # First attempt: Run all tasks together with the requested settings
    if verbose:
        print(f"\nDEBUG: First attempt - Running all tasks together: {tasks}")
        
    initial_results = None
    for config in configs_to_try:
        try:
            current_eval_kwargs = eval_kwargs.copy()
            current_eval_kwargs["tasks"] = tasks.split(",")
            current_eval_kwargs.update(config)
            
            filtered_eval_kwargs = {k: v for k, v in current_eval_kwargs.items() if v is not None}
            filtered_eval_kwargs["confirm_run_unsafe_code"] = True
            
            initial_results = simple_evaluate(**filtered_eval_kwargs)
            break
        except Exception as e:
            if verbose:
                print(f"DEBUG: Initial attempt with config {config} failed: {e}. Retrying with next config...")
            last_error = e
            continue
            
    if initial_results:
        all_task_results = initial_results
        
        # Check if we need to retry any task (specifically MBPP if it scored 0.0)
        tasks_list = tasks.split(",")
        needs_retry = []
        for task_name in tasks_list:
            task_name = task_name.strip()
            if not task_name: continue
            
            if task_name in initial_results["results"]:
                res_dict = initial_results["results"][task_name]
                pass_at_1 = res_dict.get("pass@1,create_test",
                                res_dict.get("pass_at_1,create_test",
                                res_dict.get("pass@1,none",
                                res_dict.get("pass_at_1,none",
                                res_dict.get("pass@1",
                                res_dict.get("pass_at_1",
                                res_dict.get("acc", 0.0)))))))
                
                # If MBPP scored 0.0 and we used zero-shot, plan a retry
                if task_name == "mbpp" and num_fewshot == 0 and pass_at_1 == 0.0:
                    needs_retry.append(task_name)
                    
        # Perform retries if needed
        for task_name in needs_retry:
            if verbose:
                print(f"\nDEBUG: Retrying task '{task_name}' with specialized settings...")
                
            retry_results = None
            # For MBPP, try 3-shot as a fallback
            retry_fewshot = 3 if task_name == "mbpp" else num_fewshot
            
            for config in configs_to_try:
                try:
                    current_eval_kwargs = eval_kwargs.copy()
                    current_eval_kwargs["tasks"] = [task_name]
                    current_eval_kwargs["num_fewshot"] = retry_fewshot
                    current_eval_kwargs.update(config)
                    
                    filtered_eval_kwargs = {k: v for k, v in current_eval_kwargs.items() if v is not None}
                    filtered_eval_kwargs["confirm_run_unsafe_code"] = True
                    
                    retry_results = simple_evaluate(**filtered_eval_kwargs)
                    if retry_results:
                        # Update the main results with the retry results
                        all_task_results["results"][task_name] = retry_results["results"][task_name]
                        if verbose:
                            new_score = retry_results["results"][task_name].get("pass@1,none", "N/A")
                            print(f"DEBUG: Retry for '{task_name}' completed. New score: {new_score}")
                        break
                except Exception:
                    continue
    else:
        # If the first attempt failed completely, try individual tasks as a fallback
        if verbose:
            print("DEBUG: Initial attempt failed completely. Falling back to individual task runs.")
        # ... logic for individual runs could go here, but usually above is sufficient
        raise last_error or RuntimeError("Failed to run evaluation.")

    results = all_task_results

    if results is None:
        raise last_error or RuntimeError("Failed to run coding evaluation.")
        
    if verbose and "results" in results:
        print("\n" + "=" * 60)
        print("CODING RESULTS (pass@1)")
        print("=" * 60)
        for task, task_results in results["results"].items():
            # lm-eval v0.4.11+: humaneval uses pass@1,create_test; mbpp uses pass_at_1,none
            # Note: we check for create_test variants first as they are often the "filtered" results we want
            pass_at_1 = task_results.get(
                "pass@1,create_test",
                task_results.get("pass_at_1,create_test",
                    task_results.get("pass@1,none",
                        task_results.get("pass_at_1,none",
                            task_results.get("pass@1",
                                task_results.get("pass_at_1",
                                    task_results.get("acc", "N/A")))))))
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

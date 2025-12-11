"""
MATH benchmark evaluation harness.
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
    print("Warning: lm-evaluation-harness not installed. Install with: pip install lm-eval")


def evaluate_math(
    model_path: str,
    num_fewshot: int = 4,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 8,
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on the MATH benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        num_fewshot: Number of few-shot examples (default: 4)
        limit: Limit number of examples (None = all)
        device: Device to run on (auto-detect if None)
        dtype: Model dtype (auto-detect if None)
        batch_size: Batch size for evaluation
        trust_remote_code: Whether to trust remote code
        apply_chat_template: Whether to apply the model's chat template (auto for instruct/chat if None)
        
    Returns:
        Dictionary with evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-evaluation-harness is required for MATH evaluation. "
            "Install with: pip install lm-eval"
        )
    
    if verbose:
        print("=" * 60)
        print("MATH EVALUATION")
        print("=" * 60)
    
    # Auto-detect dtype if not specified
    if dtype is None:
        if torch.cuda.is_available():
            dtype_str = "float16"
        elif torch.backends.mps.is_available():
            dtype_str = "float16"
        else:
            dtype_str = "float32"
    else:
        dtype_str = str(dtype).replace("torch.", "")
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Auto-apply chat templates for instruct/chat models if not explicitly set
    if apply_chat_template is None:
        lower_path = model_path.lower()
        # Check if it's an instruct/chat model
        apply_chat_template = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int", "llama-3"])

    # Convert to absolute path if it's a local path (for lm-eval compatibility)
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    
    # Build model_args string for lm-eval
    # lm-eval's from_pretrained will automatically detect and use .safetensors files
    base_model_args_parts = [f"pretrained={model_path}", f"dtype={dtype_str}"]
    if trust_remote_code:
        base_model_args_parts.append("trust_remote_code=True")
    base_model_args_str = ",".join(base_model_args_parts)
    
    # Run evaluation using lm-evaluation-harness
    # Note: simple_evaluate will load the model internally
    if verbose:
        print(f"\nRunning MATH evaluation with {num_fewshot}-shot learning...")
        print(f"Model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype_str}")
        if limit:
            print(f"Limiting to {limit} examples")
    else:
        print("MATH: Running...", end=" ", flush=True)
    
    # Try common task aliases in case the installed lm-eval version uses different names
    task_candidates = ["math", "hendrycks_math500"]
    results = None
    task_errors = []
    
    import logging
    lm_eval_logger = logging.getLogger("lm_eval")
    old_level = lm_eval_logger.level
    if not verbose:
        lm_eval_logger.setLevel(logging.WARNING)

    # Try different configurations for chat template
    # apply_chat_template should be passed as a parameter to simple_evaluate, not in model_args
    configs_to_try = []
    if apply_chat_template:
        # Try with chat template and fewshot_as_multiturn first (best for instruct models)
        configs_to_try.append({
            "apply_chat_template": True,
            "fewshot_as_multiturn": True
        })
        # Try with just chat template
        configs_to_try.append({
            "apply_chat_template": True
        })
    # Always include base (without chat template) as fallback
    configs_to_try.append({})
    
    for task_name in task_candidates:
        for config in configs_to_try:
            try:
                # Suppress the chat template warnings - they're just informational
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*chat template.*")
                    
                    # Build kwargs for simple_evaluate
                    eval_kwargs = {
                        "model": "hf",
                        "model_args": base_model_args_str,
                        "tasks": task_name,
                        "num_fewshot": num_fewshot,
                        "limit": limit,
                        "batch_size": batch_size,
                        "device": device,
                    }
                    # Add chat template config if specified
                    eval_kwargs.update(config)
                    
                    results = simple_evaluate(**eval_kwargs)
                break
            except TypeError as e:
                if "apply_chat_template" in str(e) or "fewshot_as_multiturn" in str(e):
                    if verbose:
                        print(
                            f"Chat template config not supported: {config}; "
                            "retrying without it."
                        )
                    continue
                raise
            except Exception as e:
                # Fall back only when the task name is missing in this lm-eval build
                if isinstance(e, KeyError) or task_name in str(e) or "not found" in str(e).lower():
                    task_errors.append(str(e))
                    if verbose:
                        print(f"Task '{task_name}' not found, trying next alias if available...")
                    break
                raise
        if results is not None:
            break
    
    if not verbose:
        lm_eval_logger.setLevel(old_level)
    
    if results is None:
        raise RuntimeError(
            f"Failed to run MATH; tried aliases {task_candidates}. Errors: {task_errors}"
        )
    
    # Extract and print key metrics
    # lm-eval results structure can vary: 
    # {"results": {"task_name": {"acc,none": value, ...}}, ...}
    # or {"results": {"task_name": {"acc": value, ...}}, ...}
    math_score = None
    task_key_used = None
    
    if "results" in results:
        math_results = results["results"]
        # Try different possible keys for MATH results
        for key in ["math", "hendrycks_math500", "hendrycks_math", "hendrycksMath"]:
            if key in math_results:
                task_result = math_results[key]
                if isinstance(task_result, dict):
                    # Try different accuracy key formats (order matters - try most specific first)
                    for acc_key in ["acc,none", "acc", "accuracy", "exact_match,none", "exact_match"]:
                        if acc_key in task_result:
                            math_score = task_result[acc_key]
                            task_key_used = key
                            break
                    if math_score is not None:
                        break
        
        # If still not found, search for any key containing "math"
        if math_score is None:
            for key, value in math_results.items():
                if "math" in key.lower() and isinstance(value, dict):
                    for acc_key in ["acc,none", "acc", "accuracy", "exact_match,none", "exact_match"]:
                        if acc_key in value:
                            math_score = value[acc_key]
                            task_key_used = key
                            break
                    if math_score is not None:
                        break
    
    if math_score is not None:
        if verbose:
            print("\n" + "=" * 60)
            print("MATH RESULTS")
            print("=" * 60)
            print(f"\nMATH Accuracy: {math_score:.4f}")
            if task_key_used:
                print(f"Task key: {task_key_used}")
        else:
            print(f"Accuracy: {math_score:.4f}")
    else:
        # Debug: print what we actually got
        print("\n" + "=" * 60)
        print("MATH RESULTS - EXTRACTION FAILED")
        print("=" * 60)
        print(f"Results type: {type(results)}")
        if isinstance(results, dict):
            print(f"Top-level keys: {list(results.keys())}")
            if "results" in results:
                print(f"Results['results'] type: {type(results['results'])}")
                if isinstance(results['results'], dict):
                    print(f"Results['results'] keys: {list(results['results'].keys())}")
                    # Print first few result entries for debugging
                    for key, value in list(results['results'].items())[:3]:
                        print(f"  {key}: {type(value)} = {value}")
        print("\nFull results structure (first 500 chars):")
        import json
        print(json.dumps(results, indent=2, default=str)[:500])
        print("\nERROR: Could not extract accuracy from results.")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on MATH benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HuggingFace ID or local path)")
    parser.add_argument("--num_fewshot", type=int, default=4,
                        help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (auto-detect if not specified)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code in model config")
    
    args = parser.parse_args()
    
    results = evaluate_math(
        model_path=args.model_path,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

"""
MATH benchmark evaluation harness.
"""

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


def evaluate_math(
    model_path: str,
    model: str = "hf",
    num_fewshot: int = 4,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: Any = "auto",
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on the MATH benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        model: Model backend ("hf" or "vllm")
        num_fewshot: Number of few-shot examples (default: 4)
        limit: Limit number of examples (None = all)
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
            "lm-evaluation-harness is required for MATH evaluation. "
            "Install with: pip install lm-eval"
        )
    
    if verbose:
        print("=" * 60)
        print("MATH EVALUATION")
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
        raise ImportError("lm-evaluation-harness is required for MATH evaluation.")
        
    # Build model_args string for lm-eval
    # lm-eval's from_pretrained will automatically detect and use .safetensors files
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
        # Limit max_num_seqs and memory to avoid OOM
        base_model_args_parts.append("max_num_seqs=32")
        base_model_args_parts.append("gpu_memory_utilization=0.7")
        
    base_model_args_str = ",".join(base_model_args_parts)
    
    # Run evaluation using lm-evaluation-harness
    # Note: simple_evaluate will load the model internally
    if verbose:
        print(f"\nRunning MATH evaluation with {num_fewshot}-shot learning...")
        print(f"Model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype_str}")
        print(f"Chat template: {apply_chat_template}")
        print(f"Generation params: temperature=0.0, max_gen_toks=512")
        if limit:
            print(f"Limiting to {limit} examples")
    else:
        print("MATH: Running...", end=" ", flush=True)
    
    # Try common task aliases - hendrycks_math is preferred in 0.4.x
    task_candidates = ["hendrycks_math", "math", "minerva_math"]
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
                        "model": model,
                        "model_args": base_model_args_str,
                        "tasks": task_name,
                        "num_fewshot": num_fewshot,
                        "limit": limit,
                        "batch_size": batch_size,
                        # Set generation parameters for proper evaluation
                        "gen_kwargs": {
                            "temperature": 0.0,  # Deterministic for fair evaluation
                            "max_gen_toks": 512,  # Sufficient for MATH answers
                        }
                    }
                    # Only pass device for non-vllm models (vllm handles devices internally)
                    if model != "vllm":
                        eval_kwargs["device"] = device
                        
                    # Add chat template config if specified
                    eval_kwargs.update(config)
                    
                    # Filter out None values to prevent library-level crashes on comparisons
                    filtered_eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
                    
                    results = simple_evaluate(**filtered_eval_kwargs)
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                # Check for known chat template or fewshot issues
                error_msg = str(e)
                chat_template_errors = [
                    "apply_chat_template",
                    "fewshot_as_multiturn",
                    "Answer is not a string",
                    "AssertionError"
                ]
                
                if any(err in error_msg or (isinstance(e, AssertionError) and "Answer is not a string" in error_msg) for err in chat_template_errors) or isinstance(e, AssertionError):
                    if verbose:
                        print(
                            f"Chat template config or mult-turn fewshot not supported: {config}; "
                            "retrying with simpler config."
                        )
                    continue
                
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
    math_score = None
    task_key_used = None
    acc_key_used = None
    
    if "results" in results:
        math_results = results["results"]
        
        # Try different possible keys for MATH results
        # We prioritize the exact task names we requested
        for key in ["math", "hendrycks_math"]:
            if key in math_results:
                task_result = math_results[key]
                if isinstance(task_result, dict):
                    # Try different accuracy key formats (order matters - try most specific first)
                    for acc_key in ["acc,none", "acc", "exact_match,none", "exact_match"]:
                        if acc_key in task_result:
                            math_score = task_result[acc_key]
                            if isinstance(math_score, (int, float)):
                                task_key_used = key
                                acc_key_used = acc_key
                                break
                    if math_score is not None:
                        break
        
        # If still not found, search for any key containing "math" (handles nested keys like "math/test")
        if math_score is None:
            for key, value in math_results.items():
                if "math" in key.lower() and isinstance(value, dict):
                    for acc_key in ["acc,none", "acc", "exact_match,none", "exact_match"]:
                        if acc_key in value:
                            math_score = value[acc_key]
                            if isinstance(math_score, (int, float)):
                                task_key_used = key
                                acc_key_used = acc_key
                                break
                    if math_score is not None:
                        break
    
    if math_score is not None:
        if verbose:
            print("\n" + "=" * 60)
            print("MATH RESULTS")
            print("=" * 60)
            print(f"\nMATH Accuracy: {math_score:.4f}")
            print(f"Task key: {task_key_used}")
            print(f"Accuracy key: {acc_key_used}")
        else:
            print(f"Accuracy: {math_score:.4f}")
    else:
        # Only print the full structure if specifically requested or if extraction fails
        print("\n" + "=" * 60)
        print("MATH RESULTS - EXTRACTION FAILED")
        print("=" * 60)
        if isinstance(results, dict) and "results" in results:
            print(f"Results['results'] keys: {list(results['results'].keys())}")
            for key, value in results['results'].items():
                if isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())}")
        print("\nERROR: Could not extract accuracy from results.")
    
    return results
    
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

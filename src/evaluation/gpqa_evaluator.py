"""
GPQA Diamond benchmark evaluation harness.
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


def evaluate_gpqa_diamond(
    model_path: str,
    model: str = "hf",
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 8,
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on the GPQA Diamond benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        num_fewshot: Number of few-shot examples (default: 0)
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
            "lm-evaluation-harness is required for GPQA Diamond evaluation. "
            "Install with: pip install lm-eval"
        )
    
    if verbose:
        print("=" * 60)
        print("GPQA EVALUATION")
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
    
    # Build model_args string for lm-eval
    # lm-eval's from_pretrained will automatically detect and use .safetensors files
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
        # GPQA is very memory intensive due to long contexts/logprobs
        base_model_args_parts.append("max_num_seqs=16")
        base_model_args_parts.append("gpu_memory_utilization=0.6")
        
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
        print(f"\nRunning GPQA Diamond evaluation with {num_fewshot}-shot learning...")
        print(f"Model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype_str}")
        if limit:
            print(f"Limiting to {limit} examples")
    else:
        print("GPQA: Running...", end=" ", flush=True)
    
    # Task name varies across lm-eval versions
    # We add 'gpqa' as a generic fallback which might be a group containing diamond
    # Try common task aliases
    # gpqa_diamond_zeroshot was the winner in previous runs
    task_candidates = ["gpqa_diamond_zeroshot", "gpqa_diamond", "gpqa_diamond_n-shot", "gpqa"]
    results = None
    task_errors = []
    
    import logging
    lm_eval_logger = logging.getLogger("lm_eval")
    old_level = lm_eval_logger.level
    if not verbose:
        lm_eval_logger.setLevel(logging.WARNING)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*chat template.*")
        
        for task_name in task_candidates:
            for config in configs_to_try:
                try:
                    eval_kwargs = {
                        "model": model, # Use passed-in backend
                        "model_args": base_model_args_str,
                        "tasks": task_name,
                        "num_fewshot": num_fewshot,
                        "limit": limit,
                        "batch_size": batch_size,
                        # Set generation parameters for proper evaluation
                        "gen_kwargs": {
                            "temperature": 0.0,  # Deterministic for fair evaluation
                            "max_gen_toks": 256,  # Sufficient for GPQA answers
                        }
                    }
                    # Only pass device for non-vllm models (vllm handles devices internally)
                    if model != "vllm":
                        eval_kwargs["device"] = device
                        
                    eval_kwargs.update(config)
                    
                    # Filter out None values to prevent library-level crashes on comparisons
                    filtered_eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
                    
                    if verbose:
                        print(f"DEBUG: Attempting task '{task_name}' with config {config}")
                        
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
                        continue
                        
                    if isinstance(e, KeyError) or task_name in str(e) or "not found" in str(e).lower():
                        task_errors.append(f"{task_name}: {str(e)}")
                        if verbose:
                            print(f"Task '{task_name}' not found or failed, trying next alias/config...")
                        break
                    
                    # Print full traceback for unexpected errors
                    traceback.print_exc()
                    raise
            if results is not None:
                break
    
    if not verbose:
        lm_eval_logger.setLevel(old_level)
    
    if results is None:
        raise RuntimeError(
            f"Failed to run GPQA; tried aliases {task_candidates}. Errors: {task_errors}"
        )
    
    # Extract and print key metrics
    if "results" in results:
        gpqa_results = results["results"]
        if verbose:
             print(f"DEBUG: GPQA Raw results keys: {list(gpqa_results.keys())}")
             
        found_key = False
        for key, value in gpqa_results.items():
            if "gpqa" in key.lower() or "diamond" in key.lower():
                found_key = True
                if isinstance(value, dict):
                    gpqa_score = value.get("acc_norm,none", value.get("acc_norm", value.get("acc,none", value.get("acc", 0))))
                    if verbose:
                        print("\n" + "=" * 60)
                        print("GPQA DIAMOND RESULTS")
                        print("=" * 60)
                        print(f"\nGPQA Diamond Accuracy: {gpqa_score:.4f}")
                        print(f"Available metrics for this task: {list(value.keys())}")
                    else:
                        print(f"GPQA Diamond Accuracy: {gpqa_score:.4f}")
                break
        
        if not found_key and verbose:
            print(f"WARNING: Could not find any keys containing 'gpqa' or 'diamond' in results: {list(gpqa_results.keys())}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on GPQA Diamond benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HuggingFace ID or local path)")
    parser.add_argument("--num_fewshot", type=int, default=0,
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
    
    results = evaluate_gpqa_diamond(
        model_path=args.model_path,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

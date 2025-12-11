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
        apply_chat_template = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])

    # Build model_args string for lm-eval
    base_model_args_parts = [f"pretrained={model_path}", f"dtype={dtype_str}"]
    if trust_remote_code:
        base_model_args_parts.append("trust_remote_code=True")
    base_model_args_str = ",".join(base_model_args_parts)

    model_args_variants = []
    if apply_chat_template:
        model_args_variants.append(base_model_args_str + ",apply_chat_template=True")
    model_args_variants.append(base_model_args_str)
    
    # Run evaluation using lm-evaluation-harness
    # Note: simple_evaluate will load the model internally
    print(f"\nRunning MATH evaluation with {num_fewshot}-shot learning...")
    print(f"Model: {model_path}")
    print(f"Device: {device}, Dtype: {dtype_str}")
    if limit:
        print(f"Limiting to {limit} examples")
    
    # Try common task aliases in case the installed lm-eval version uses different names
    task_candidates = ["math", "hendrycks_math500"]
    results = None
    task_errors = []

    for task_name in task_candidates:
        for model_args_str in model_args_variants:
            try:
                results = simple_evaluate(
                    model="hf",
                    model_args=model_args_str,
                    tasks=task_name,
                    num_fewshot=num_fewshot,
                    limit=limit,
                    batch_size=batch_size,
                    device=device,
                )
                break
            except TypeError as e:
                if "apply_chat_template" in str(e):
                    print(
                        "apply_chat_template not supported by this lm-eval/transformers version; "
                        "retrying without it."
                    )
                    continue
                raise
            except Exception as e:
                # Fall back only when the task name is missing in this lm-eval build
                if isinstance(e, KeyError) or task_name in str(e) or "not found" in str(e).lower():
                    task_errors.append(str(e))
                    print(f"Task '{task_name}' not found, trying next alias if available...")
                    break
                raise
        if results is not None:
            break
    if results is None:
        raise RuntimeError(
            f"Failed to run MATH; tried aliases {task_candidates}. Errors: {task_errors}"
        )
    
    print("\n" + "=" * 60)
    print("MATH RESULTS")
    print("=" * 60)
    
    # Extract and print key metrics
    if "results" in results:
        math_results = results["results"]
        if "math" in math_results:
            math_score = math_results["math"].get("acc,none", 0)
            print(f"\nMATH Accuracy: {math_score:.4f}")
        else:
            # Check for other possible keys
            for key, value in math_results.items():
                if "math" in key.lower():
                    if "acc" in value:
                        print(f"\nMATH Accuracy: {value['acc']:.4f}")
                    break
    
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

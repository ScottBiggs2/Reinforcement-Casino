"""
MMLU (Massive Multitask Language Understanding) evaluation harness.
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


def evaluate_mmlu(
    model_path: str,
    num_fewshot: int = 5,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 8,
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on the MMLU benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        num_fewshot: Number of few-shot examples (default: 5)
        limit: Limit number of examples per task (None = all)
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
            "lm-evaluation-harness is required for MMLU evaluation. "
            "Install with: pip install lm-eval"
        )
    
    if verbose:
        print("=" * 60)
        print("MMLU EVALUATION")
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
    base_model_args_parts = [f"pretrained={model_path}", f"dtype={dtype_str}"]
    if trust_remote_code:
        base_model_args_parts.append("trust_remote_code=True")
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
                    "model": "hf",
                    "model_args": base_model_args_str,
                    "tasks": "mmlu",
                    "num_fewshot": num_fewshot,
                    "limit": limit,
                    "batch_size": batch_size,
                    "device": device,
                    # Set generation parameters for proper evaluation
                    "gen_kwargs": {
                        "temperature": 0.0,  # Deterministic for fair evaluation
                        "max_gen_toks": 256,  # Sufficient for MMLU answers
                    }
                }
                eval_kwargs.update(config)
                results = simple_evaluate(**eval_kwargs)
                break
            except TypeError as e:
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

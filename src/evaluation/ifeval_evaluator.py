"""
IFEval (Instruction Following Evaluation) benchmark evaluation harness.
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


def evaluate_ifeval(
    model_path: str,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on the IFEval benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        limit: Limit number of examples (None = all)
        device: Device to run on
        dtype: Model dtype
        batch_size: Batch size
        trust_remote_code: Whether to trust remote code
        apply_chat_template: Whether to apply the model's chat template
        
    Returns:
        Dictionary with evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError("lm-evaluation-harness is required for IFEval evaluation.")
    
    if verbose:
        print("=" * 60)
        print("IFEVAL EVALUATION")
        print("=" * 60)
    
    # Auto-detect dtype
    if dtype is None:
        dtype_str = "float16" if torch.cuda.is_available() else "float32"
    else:
        dtype_str = str(dtype).replace("torch.", "")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    base_model_args_str = f"pretrained={model_path},dtype={dtype_str}"
    if trust_remote_code:
        base_model_args_str += ",trust_remote_code=True"
    
    if verbose:
        print(f"\nRunning IFEval evaluation...")
        print(f"Model: {model_path}")
        print(f"Chat template: {apply_chat_template}")
    
    eval_kwargs = {
        "model": "hf",
        "model_args": base_model_args_str,
        "tasks": ["ifeval"],
        "num_fewshot": 0, # IFEval is 0-shot
        "limit": limit,
        "batch_size": batch_size,
        "device": device,
        "gen_kwargs": {
            "temperature": 0.0,
            "max_gen_toks": 1024, # IFEval responses can be long
        }
    }
    
    if apply_chat_template:
        eval_kwargs["apply_chat_template"] = True
        
    results = simple_evaluate(**eval_kwargs)
    
    if verbose and "results" in results and "ifeval" in results["results"]:
        ifeval_data = results["results"]["ifeval"]
        print("\n" + "=" * 60)
        print("IFEVAL RESULTS")
        print("=" * 60)
        # IFEval has strict and loose accuracy
        strict_acc = ifeval_data.get("prompt_level_strict_acc,none", ifeval_data.get("prompt_level_strict_acc", "N/A"))
        loose_acc = ifeval_data.get("prompt_level_loose_acc,none", ifeval_data.get("prompt_level_loose_acc", "N/A"))
        print(f"Prompt-level Strict Acc: {strict_acc}")
        print(f"Prompt-level Loose Acc: {loose_acc}")
            
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on IFEval")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    
    args = parser.parse_args()
    evaluate_ifeval(
        model_path=args.model_path,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )

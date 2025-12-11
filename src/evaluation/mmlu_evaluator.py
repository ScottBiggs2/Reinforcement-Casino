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


def evaluate_mmlu(
    model_path: str,
    num_fewshot: int = 5,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 8,
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
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
    if apply_chat_template is None:
        lower_path = model_path.lower()
        apply_chat_template = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])

    # Build model_args string for lm-eval
    model_args_parts = [f"pretrained={model_path}", f"dtype={dtype_str}"]
    if trust_remote_code:
        model_args_parts.append("trust_remote_code=True")
    if apply_chat_template:
        model_args_parts.append("apply_chat_template=True")
    model_args_str = ",".join(model_args_parts)
    
    # Run evaluation using lm-evaluation-harness
    # Note: simple_evaluate will load the model internally
    print(f"\nRunning MMLU evaluation with {num_fewshot}-shot learning...")
    print(f"Model: {model_path}")
    print(f"Device: {device}, Dtype: {dtype_str}")
    if limit:
        print(f"Limiting to {limit} examples per task")
    
    results = simple_evaluate(
        model="hf",
        model_args=model_args_str,
        tasks="mmlu",
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        device=device,
    )
    
    print("\n" + "=" * 60)
    print("MMLU RESULTS")
    print("=" * 60)
    
    # Extract and print key metrics
    if "results" in results:
        mmlu_results = results["results"]
        if "mmlu" in mmlu_results:
            mmlu_score = mmlu_results["mmlu"].get("acc,none", 0)
            print(f"\nMMLU Accuracy: {mmlu_score:.4f}")
        else:
            # MMLU has multiple subtasks
            subtask_scores = {}
            for key, value in mmlu_results.items():
                if "mmlu" in key.lower():
                    if "acc" in value:
                        subtask_scores[key] = value["acc"]
            if subtask_scores:
                avg_score = sum(subtask_scores.values()) / len(subtask_scores)
                print(f"\nMMLU Average Accuracy: {avg_score:.4f}")
                print("\nPer-subject scores:")
                for subject, score in sorted(subtask_scores.items()):
                    print(f"  {subject}: {score:.4f}")
    
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

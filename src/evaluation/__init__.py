"""
LLM Evaluation Harnesses

This package provides evaluation harnesses for common LLM benchmarks:
- MMLU (Massive Multitask Language Understanding)
- MATH
- SQuAD (Stanford Question Answering Dataset)
- GPQA Diamond

All evaluators support loading models from:
- HuggingFace model IDs (e.g., "google/gemma-2-2b")
- Local paths containing .safetensors files
- Local checkpoint directories
"""

from evaluation.model_loader import load_model_and_tokenizer
from evaluation.mmlu_evaluator import evaluate_mmlu
from evaluation.math_evaluator import evaluate_math
from evaluation.squad_evaluator import evaluate_squad
from evaluation.gpqa_evaluator import evaluate_gpqa_diamond

__all__ = [
    "load_model_and_tokenizer",
    "evaluate_mmlu",
    "evaluate_math",
    "evaluate_squad",
    "evaluate_gpqa_diamond",
]

"""Training AMP / model dtype selection for GRPO (bf16 vs fp16 vs auto)."""

from __future__ import annotations

from typing import Tuple

import torch


def resolve_grpo_precision(mode: str = "auto") -> Tuple[bool, bool, torch.dtype]:
    """
    Returns (bf16, fp16, model_torch_dtype) for HuggingFace TrainingArguments + from_pretrained.

    - auto: bf16 when ``torch.cuda.is_bf16_supported()`` (typical Ampere+); else fp16 on GPU.
    - bf16 / fp16: force that path (may raise from Trainer if unsupported).
    """
    if mode not in ("auto", "bf16", "fp16"):
        mode = "auto"
    if mode == "bf16":
        return True, False, torch.bfloat16
    if mode == "fp16":
        return False, True, torch.float16
    if not torch.cuda.is_available():
        return False, False, torch.float32
    if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return True, False, torch.bfloat16
    return False, True, torch.float16

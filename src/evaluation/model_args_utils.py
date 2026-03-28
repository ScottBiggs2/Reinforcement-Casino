import os
from typing import List


def build_lm_eval_model_args_parts(
    *,
    model_path: str,
    dtype_str: str,
    backend: str,
    trust_remote_code: bool,
) -> List[str]:
    """
    Build model_args parts for lm-eval.

    For local checkpoints on the Hugging Face backend, avoid passing an explicit
    dtype through model_args. Some transformers/lm-eval combinations feed that
    dtype into AutoConfig.from_pretrained, which can trigger config repr/json
    failures on torch.dtype objects for locally saved checkpoints.
    """
    parts = [f"pretrained={model_path}"]

    is_local_path = os.path.exists(model_path)
    if not (backend == "hf" and is_local_path):
        parts.append(f"dtype={dtype_str}")

    if trust_remote_code:
        parts.append("trust_remote_code=True")

    return parts

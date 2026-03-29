import os
from typing import Optional


def has_chat_template(model_path: str) -> bool:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer.chat_template is not None
    except Exception:
        return False


def auto_detect_apply_chat_template(
    model_path: str,
    *,
    explicit_value: Optional[bool],
    verbose: bool = False,
) -> bool:
    """
    Decide whether lm-eval should apply a chat template.

    Hugging Face model IDs can use path-name heuristics (`instruct`, `chat`, `-it`)
    because the remote tokenizer/config is the source of truth.

    Local checkpoints are stricter: only enable chat templating if the local tokenizer
    actually exposes `tokenizer.chat_template`. Fine-tuned checkpoints often live in
    paths containing `instruct`, but may not carry tokenizer assets or a chat template.
    """
    if explicit_value is not None:
        if verbose:
            print(f"Chat template explicitly set to: {explicit_value}")
        return explicit_value

    lower_path = model_path.lower()
    path_has_instruct = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])
    is_local_path = os.path.exists(model_path)

    if is_local_path:
        local_has_template = has_chat_template(model_path)
        if local_has_template:
            if verbose:
                print("✓ Local tokenizer exposes chat_template; enabling chat template")
            return True
        if verbose:
            if path_has_instruct:
                print("⚠ Local checkpoint path looks instruct-tuned, but tokenizer.chat_template is missing; disabling chat template")
            else:
                print("⚠ Local checkpoint tokenizer has no chat_template; disabling chat template")
        return False

    if verbose:
        if path_has_instruct:
            print("✓ Detected instruct model from HuggingFace path, enabling chat template")
        else:
            print("⚠ HuggingFace model path doesn't indicate instruct model, disabling chat template")
    return path_has_instruct

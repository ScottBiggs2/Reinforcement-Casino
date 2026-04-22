import os
from pathlib import Path
from typing import Any, Optional, Tuple


def _materialize_chat_template(tokenizer: Any, model_path: str) -> Tuple[Optional[str], str]:
    """
    Return a concrete chat-template string if available.

    Transformers may represent chat templates as:
    - a string (ideal)
    - a dict-like (multiple templates)
    - absent on local checkpoints even when `chat_template.jinja` exists
    """
    try:
        raw = getattr(tokenizer, "chat_template", None)
    except Exception:
        raw = None

    if isinstance(raw, str) and raw.strip():
        return raw, "tokenizer.chat_template (string)"

    # Some versions expose multiple templates (dict-like). Pick deterministically.
    if isinstance(raw, dict) and raw:
        for key in ("default", "chat", "llama-3", "llama3", "instruct"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val, f"tokenizer.chat_template[{key!r}]"
        # Fall back to first non-empty string value.
        for key, val in raw.items():
            if isinstance(val, str) and val.strip():
                return val, f"tokenizer.chat_template[{key!r}]"

    # Local checkpoints often have a `chat_template.jinja` sidecar.
    try:
        template_path = Path(model_path) / "chat_template.jinja"
        if template_path.exists():
            text = template_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text, "chat_template.jinja"
    except Exception:
        pass

    return None, "no_chat_template"


def has_chat_template(model_path: str) -> bool:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        template, _ = _materialize_chat_template(tokenizer, model_path)
        return template is not None
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
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            template, source = _materialize_chat_template(tokenizer, model_path)
        except Exception:
            template, source = None, "tokenizer_load_failed"

        if template is not None:
            if verbose:
                print(f"✓ Local checkpoint has chat template ({source}); enabling chat template")
            return True

        if verbose:
            if path_has_instruct:
                print(
                    "⚠ Local checkpoint path looks instruct-tuned, but no usable chat template was found "
                    f"({source}); disabling chat template"
                )
            else:
                print(f"⚠ Local checkpoint has no usable chat template ({source}); disabling chat template")
        return False

    if verbose:
        if path_has_instruct:
            print("✓ Detected instruct model from HuggingFace path, enabling chat template")
        else:
            print("⚠ HuggingFace model path doesn't indicate instruct model, disabling chat template")
    return path_has_instruct

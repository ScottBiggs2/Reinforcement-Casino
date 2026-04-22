import json
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


def _persist_chat_template_to_tokenizer_config(model_path: str, template: str, *, verbose: bool) -> bool:
    """
    Ensure Transformers loads `tokenizer.chat_template` from disk.

    Some training checkpoints ship `chat_template.jinja` but do not embed the template into
    `tokenizer_config.json`. lm-eval's vLLM path calls `tokenizer.apply_chat_template(...)`
    which requires `tokenizer.chat_template` to be set unless an explicit template is passed.
    """
    cfg_path = Path(model_path) / "tokenizer_config.json"
    if not cfg_path.exists():
        if verbose:
            print(f"⚠ Cannot persist chat template: missing {cfg_path}")
        return False

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        if verbose:
            print(f"⚠ Cannot read tokenizer_config.json for chat template persistence: {e}")
        return False

    existing = cfg.get("chat_template")
    if isinstance(existing, str) and existing.strip():
        return True

    cfg["chat_template"] = template
    try:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        if verbose:
            print(f"⚠ Failed writing chat_template into tokenizer_config.json: {e}")
        return False

    if verbose:
        print(f"✓ Wrote chat_template into {cfg_path} (was missing; sourced from checkpoint template files)")
    return True


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

    Local checkpoints: enable chat templating when we can materialize a concrete template
    string (tokenizer field, dict templates, or `chat_template.jinja`). If the template
    exists only as a sidecar file, we persist it into `tokenizer_config.json` so downstream
    libraries that call `tokenizer.apply_chat_template()` do not crash.
    """
    lower_path = model_path.lower()
    path_has_instruct = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])
    is_local_path = os.path.exists(model_path)

    # Explicit enablement still needs a real `tokenizer.chat_template` after `from_pretrained`
    # for lm-eval/vLLM, so do the same persistence prep as auto-detect for local dirs.
    if explicit_value is True and is_local_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            template, source = _materialize_chat_template(tokenizer, model_path)
        except Exception:
            template, source = None, "tokenizer_load_failed"

        if template is None:
            if verbose:
                print(
                    f"⚠ --apply_chat_template requested, but no usable chat template was found for local path "
                    f"({source}); refusing to enable (would crash in lm-eval)."
                )
            return False

        raw_on_tok = getattr(tokenizer, "chat_template", None)
        needs_persist = not (isinstance(raw_on_tok, str) and raw_on_tok.strip())
        if needs_persist:
            ok = _persist_chat_template_to_tokenizer_config(model_path, template, verbose=verbose)
            if not ok:
                if verbose:
                    print(
                        "⚠ --apply_chat_template requested, but chat template could not be persisted into "
                        "tokenizer_config.json; refusing to enable (would crash in lm-eval)."
                    )
                return False

        if verbose:
            print(f"Chat template explicitly set to: True (local; template source: {source})")
        return True

    if explicit_value is not None:
        if verbose:
            print(f"Chat template explicitly set to: {explicit_value}")
        return explicit_value

    if is_local_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            template, source = _materialize_chat_template(tokenizer, model_path)
        except Exception:
            template, source = None, "tokenizer_load_failed"

        if template is not None:
            # Critical: lm-eval vLLM calls `tokenizer.apply_chat_template` which requires
            # `tokenizer.chat_template` to be populated after `from_pretrained`.
            raw_on_tok = getattr(tokenizer, "chat_template", None)
            needs_persist = not (isinstance(raw_on_tok, str) and raw_on_tok.strip())
            if needs_persist:
                ok = _persist_chat_template_to_tokenizer_config(model_path, template, verbose=verbose)
                if not ok:
                    if verbose:
                        print(
                            "⚠ Local checkpoint has a materialized chat template, but could not persist it into "
                            "tokenizer_config.json; disabling chat template to avoid lm-eval crashes."
                        )
                    return False

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

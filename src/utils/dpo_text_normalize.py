"""Pure-text DPO row normalization (no torch). Shared by training, CKA, and registry."""

from __future__ import annotations

from typing import Any, Dict


def dpo_msg_to_text(x: Any) -> str:
    """Convert a DPO field (str, chat dict, or message list) to plain text.

    Matches training / ``load_dpo_dataset`` and handles Tulu3-style list-of-dicts
    (``content`` / ``value`` / ``role``) that a naive ``{"value": ...}``-only parser misses.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("value", "content", "text", "message"):
            if key in x:
                return str(x[key])
        return " ".join(str(v) for v in x.values() if isinstance(v, str))
    if isinstance(x, list):
        parts = []
        for m in x:
            if isinstance(m, dict):
                for key in ("value", "content", "text", "message"):
                    if key in m:
                        parts.append(str(m[key]))
                        break
            elif isinstance(m, str):
                parts.append(m)
        return "\n".join(parts)
    return str(x)


def extract_dpo_prompt(rec: dict) -> str:
    """Human/user prompt from ``conversations`` or ``prompt`` (shared with DPO training)."""
    convs = rec.get("conversations", None)
    if convs and isinstance(convs, list):
        human_parts = []
        for m in convs:
            if not isinstance(m, dict):
                continue
            role = m.get("from", m.get("role", "")).lower()
            if role not in ("human", "user", "system"):
                continue
            t = dpo_msg_to_text(m).strip()
            if t:
                human_parts.append(t)
        if human_parts:
            return "\n".join(human_parts).strip()

    prompt_raw = rec.get("prompt", "")
    if isinstance(prompt_raw, list):
        prompt_text = "\n".join(
            dpo_msg_to_text(m).strip()
            for m in prompt_raw
            if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
        ).strip()
        if prompt_text:
            return prompt_text

    return dpo_msg_to_text(prompt_raw).strip()


def normalize_dpo_record(rec: dict) -> Dict[str, str]:
    """Map one raw HF row to ``{prompt, chosen, rejected}`` text (same as training pipeline)."""
    prompt_text = extract_dpo_prompt(rec)
    chosen_text = dpo_msg_to_text(rec.get("chosen", "")).strip()
    rejected_text = dpo_msg_to_text(rec.get("rejected", "")).strip()
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }

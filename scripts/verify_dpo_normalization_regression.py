#!/usr/bin/env python3
"""
Runtime regression check: pre-refactor DPO normalization vs current ``normalize_dpo_record``.

Confirms byte parity on value-based Light-R1-style rows and documents intentional
prompt-only deltas for content-only chat turns. No network required.

Usage (from repo root):

  python scripts/verify_dpo_normalization_regression.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, List, Tuple

# Repo root = parent of scripts/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --- Legacy implementation (must match pre-refactor load_dpo_dataset inner functions) ---


def _legacy_msg_to_text(x: Any) -> str:
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


def _legacy_extract_prompt(rec: dict) -> str:
    convs = rec.get("conversations", None)
    if convs and isinstance(convs, list):
        human_parts = [
            m["value"]
            for m in convs
            if isinstance(m, dict)
            and m.get("from", m.get("role", "")).lower() in ("human", "user", "system")
            and "value" in m
        ]
        if human_parts:
            return "\n".join(human_parts).strip()

    prompt_raw = rec.get("prompt", "")
    if isinstance(prompt_raw, list):
        prompt_text = "\n".join(
            m.get("value", "")
            for m in prompt_raw
            if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
        ).strip()
        if prompt_text:
            return prompt_text

    return _legacy_msg_to_text(prompt_raw).strip()


def legacy_normalize_dpo_record(rec: dict) -> dict:
    prompt_text = _legacy_extract_prompt(rec)
    chosen_text = _legacy_msg_to_text(rec.get("chosen", "")).strip()
    rejected_text = _legacy_msg_to_text(rec.get("rejected", "")).strip()
    return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}


def main() -> int:
    sys.path.insert(0, _REPO_ROOT)
    from src.utils.dpo_text_normalize import normalize_dpo_record  # noqa: WPS433

    fixtures: List[Tuple[str, dict, str]] = [
        (
            "H1_value_conversations",
            {
                "conversations": [
                    {"from": "human", "value": "Say hi."},
                    {"from": "gpt", "value": "Hi."},
                ],
                "chosen": {"from": "gpt", "value": "Hello there."},
                "rejected": {"from": "gpt", "value": "Yo."},
            },
            "parity_expected",
        ),
        (
            "H2_string_prompt",
            {
                "prompt": "Plain string prompt",
                "chosen": "c",
                "rejected": "r",
            },
            "parity_expected",
        ),
        (
            "H5_list_prompt_value_only",
            {
                "prompt": [
                    {"from": "human", "value": "Q?"},
                    {"from": "assistant", "value": "ignore"},
                ],
                "chosen": {"value": "yes"},
                "rejected": {"value": "no"},
            },
            "parity_expected",
        ),
        (
            "H3_content_only_human_conv",
            {
                "conversations": [
                    {"role": "user", "content": "Use content key"},
                ],
                "chosen": {"content": "c"},
                "rejected": {"content": "r"},
            },
            "divergence_expected",
        ),
        (
            "H4_conversations_content_no_value",
            {
                "conversations": [
                    {"from": "human", "content": "No value key here"},
                ],
                "chosen": {"value": "c"},
                "rejected": {"value": "r"},
            },
            "divergence_expected",
        ),
    ]

    passed = 0
    failed = 0

    for name, rec, expect in fixtures:
        old = legacy_normalize_dpo_record(rec)
        new = normalize_dpo_record(rec)
        same = old == new
        chosen_same = old["chosen"] == new["chosen"]
        rej_same = old["rejected"] == new["rejected"]

        if expect == "parity_expected":
            if same:
                passed += 1
                print(f"OK  {name}: byte parity with legacy")
            else:
                failed += 1
                print(f"FAIL {name}: expected parity\n  old={old!r}\n  new={new!r}")
        else:
            if not same and chosen_same and rej_same:
                passed += 1
                print(f"OK  {name}: prompt-only divergence (documented)")
            else:
                failed += 1
                print(f"FAIL {name}: unexpected outcome\n  old={old!r}\n  new={new!r}")

    print(f"\nSummary: passed={passed} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

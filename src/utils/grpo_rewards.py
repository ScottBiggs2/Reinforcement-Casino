"""
Shared GRPO reward functions for math training (dense + sparse BSR).

Profiles:
- ``openr1_tags``: Only ``<redacted_thinking>...</redacted_thinking>`` splits thinking vs
  final answer (legacy OpenR1-style). Models that never emit these tags get empty
  ``thinking_content`` unless you add a prompt instruction (see ``grpo_prompt_suffix``).
- ``llama_cot`` (default): Try redacted tags if present; otherwise split reasoning vs answer
  using ``####``, ``\\boxed{...}``, or ``Final Answer:`` / ``Answer:`` heuristics so Instruct
  models without tags still get a meaningful format signal.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Literal, Optional

RewardProfile = Literal["openr1_tags", "llama_cot"]

_VALID_PROFILES = frozenset({"openr1_tags", "llama_cot"})
_DEFAULT_PROFILE: RewardProfile = "llama_cot"

# Minimum non-whitespace chars in the "reasoning" region to count for format_reasoning (llama_cot).
_MIN_REASONING_CHARS = 16

_REDACTED_PATTERN = re.compile(
    r"<redacted_thinking>\s*(.*?)\s*</redacted_thinking>\s*(.*)", re.DOTALL | re.IGNORECASE
)
_FINAL_ANSWER_RE = re.compile(r"(?i)final\s+answer\s*:")
_ANSWER_LINE_RE = re.compile(r"(?i)(?:^|\n)\s*answer\s*:")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def normalize_reward_profile(name: Optional[str]) -> RewardProfile:
    if name is None or str(name).strip() == "":
        return _DEFAULT_PROFILE
    key = str(name).strip().lower()
    if key not in _VALID_PROFILES:
        raise ValueError(
            f"Unknown grpo_reward_profile={name!r}. Choose one of: {sorted(_VALID_PROFILES)}"
        )
    return key  # type: ignore[return-value]


def _parse_openr1_tags_only(text: str) -> dict:
    """Legacy: only redacted tags; else full text is ``response``."""
    match = _REDACTED_PATTERN.search(text)
    if not match:
        return {"thinking_content": "", "response": text.strip()}
    return {
        "thinking_content": match.group(1).strip(),
        "response": match.group(2).strip(),
    }


def _try_redacted_tags(text: str) -> Optional[dict]:
    match = _REDACTED_PATTERN.search(text)
    if not match:
        return None
    return {
        "thinking_content": match.group(1).strip(),
        "response": match.group(2).strip(),
    }


def _split_cot_heuristic(text: str) -> dict:
    """Split CoT vs final answer without redacted tags (best-effort)."""
    t = text.strip()
    if not t:
        return {"thinking_content": "", "response": ""}

    if "####" in t:
        before, after = t.rsplit("####", 1)
        return {"thinking_content": before.strip(), "response": after.strip()}

    boxed_matches = list(_BOXED_RE.finditer(t))
    if boxed_matches:
        m = boxed_matches[-1]
        before = t[: m.start()].strip()
        inside = m.group(1).strip()
        return {"thinking_content": before, "response": inside}

    for rx in (_FINAL_ANSWER_RE, _ANSWER_LINE_RE):
        matches = list(rx.finditer(t))
        if matches:
            m = matches[-1]
            return {"thinking_content": t[: m.start()].strip(), "response": t[m.end() :].strip()}

    return {"thinking_content": "", "response": t}


def _parse_llama_cot(text: str) -> dict:
    tagged = _try_redacted_tags(text)
    if tagged is not None:
        return tagged
    return _split_cot_heuristic(text)


def _make_parse_reasoning_fn(profile: RewardProfile) -> Callable[[str], dict]:
    if profile == "openr1_tags":
        return _parse_openr1_tags_only
    return _parse_llama_cot


def get_completion_content(completion: Any) -> str:
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in completion
        )
    return str(completion)


def get_grpo_reward_funcs(profile: str = _DEFAULT_PROFILE) -> List[Callable[..., List[float]]]:
    """
    Build reward callables for the given profile. TRL calls each with
    ``(completions, solution, **kwargs)``.
    """
    prof = normalize_reward_profile(profile)
    _parse_one = _make_parse_reasoning_fn(prof)

    def parse_responses(completions: list) -> list:
        return [_parse_one(get_completion_content(c)) for c in completions]

    def accuracy_reward(completions, solution, **kwargs) -> list[float]:
        parsed = parse_responses(completions)
        rewards = []
        for r, ans in zip(parsed, solution):
            model_answer = r["response"].strip()
            ans = str(ans) if ans is not None else ""
            target_ans = ans.split("####")[1].strip() if "####" in ans else ans.strip()
            numbers = re.findall(r"-?\d+\.?\d*", model_answer.replace(",", ""))
            model_last_num = numbers[-1] if numbers else ""
            target_numbers = re.findall(r"-?\d+\.?\d*", target_ans.replace(",", ""))
            target_last_num = target_numbers[-1] if target_numbers else target_ans
            rewards.append(
                1.0
                if model_answer == target_ans
                or (model_last_num and target_last_num and model_last_num == target_last_num)
                else 0.0
            )
        return rewards

    def format_number_reward(completions, solution=None, **kwargs) -> list[float]:
        """0.5 if the final-answer slice contains at least one numeric token (last-number signal)."""
        out = []
        for r in parse_responses(completions):
            resp = r["response"].replace(",", "")
            nums = re.findall(r"-?\d+\.?\d*", resp)
            out.append(0.5 if nums else 0.0)
        return out

    def format_reasoning_reward(completions, solution=None, **kwargs) -> list[float]:
        rewards = []
        for r in parse_responses(completions):
            think = r["thinking_content"].strip()
            resp = r["response"].strip()
            if prof == "openr1_tags":
                rewards.append(0.5 if think and resp else 0.0)
            else:
                rewards.append(
                    0.5
                    if len(think) >= _MIN_REASONING_CHARS and len(resp) >= 1
                    else 0.0
                )
        return rewards

    return [accuracy_reward, format_number_reward, format_reasoning_reward]


# Optional append for ``openr1_tags`` so models are nudged to emit redacted blocks (math-220k GRPO).
OPENR1_TAG_PROMPT_SUFFIX = (
    "\n\nPut your step-by-step reasoning inside "
    "<redacted_thinking>...</redacted_thinking>, then give the final answer after the closing tag."
)

# Append for ``llama_cot``: nudge Instruct models to emit a marker the heuristic
# split can find. Without this the heuristic falls through to thinking_content=""
# and format_reasoning_reward stays 0 forever — observed in the deleted
# 200-step math-220k GRPO run.
LLAMA_COT_PROMPT_SUFFIX = (
    "\n\nThink step by step, then put your final numeric answer inside "
    "\\boxed{...} (or write `#### <answer>` on a new line)."
)

# Default import: ``llama_cot`` (delimiter-aware); matches Open-R1 + Instruct runs.
GRPO_REWARD_FUNCS: List[Callable[..., List[float]]] = get_grpo_reward_funcs(_DEFAULT_PROFILE)

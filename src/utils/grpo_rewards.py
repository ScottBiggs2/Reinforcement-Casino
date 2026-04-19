"""
Shared GRPO reward functions aligned with OpenR1-Math / CoT (`<think>...</think>`).

Used by dense and sparse GRPO training entrypoints so objectives stay consistent.
"""

from __future__ import annotations

import re
from typing import Any


def parse_reasoning_response(text: str) -> dict:
    pattern = r"<think>\s*(.*?)\s*</think>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking_content": "", "response": text}
    return {"thinking_content": match.group(1).strip(), "response": match.group(2).strip()}


def get_completion_content(completion: Any) -> str:
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in completion
        )
    return str(completion)


def parse_responses(completions: list) -> list[dict]:
    return [parse_reasoning_response(get_completion_content(c)) for c in completions]


def _extract_final_answer(text: str) -> str:
    """Extract the final numeric/short answer from a solution string.

    Tries in order:
      1. `\\boxed{...}` (OpenR1-Math-220k, MATH convention — takes last occurrence)
      2. text after `####` (GSM8K convention)
      3. whole string stripped
    """
    # Match \boxed{...} with balanced braces (one level is enough in practice).
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1].strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def accuracy_reward(completions, solution, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = []
    for r, ans in zip(parsed_responses, solution):
        model_answer = r["response"].strip()
        ans = str(ans) if ans is not None else ""
        target_ans = _extract_final_answer(ans)
        # Also try to pull a boxed answer out of the model's response when present;
        # with math CoT, the real answer is usually inside \boxed{...}.
        model_final = _extract_final_answer(model_answer) if "\\boxed" in model_answer else model_answer
        numbers = re.findall(r"-?\d+\.?\d*", model_final.replace(",", ""))
        model_last_num = numbers[-1] if numbers else ""
        target_numbers = re.findall(r"-?\d+\.?\d*", target_ans.replace(",", ""))
        target_last_num = target_numbers[-1] if target_numbers else target_ans
        rewards.append(
            1.0
            if model_final == target_ans
            or (model_last_num and target_last_num and model_last_num == target_last_num)
            else 0.0
        )
    return rewards


def format_number_reward(completions, **kwargs) -> list[float]:
    return [
        0.5 if re.findall(r"-?\d+\.?\d*", r["response"].replace(",", "")) else 0.0
        for r in parse_responses(completions)
    ]


def format_reasoning_reward(completions, **kwargs) -> list[float]:
    return [0.5 if r["thinking_content"] and r["response"] else 0.0 for r in parse_responses(completions)]


GRPO_REWARD_FUNCS = [accuracy_reward, format_number_reward, format_reasoning_reward]

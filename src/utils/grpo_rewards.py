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


def accuracy_reward(completions, solution, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = []
    for r, ans in zip(parsed_responses, solution):
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


def format_number_reward(completions, **kwargs) -> list[float]:
    return [
        0.5 if re.findall(r"-?\d+\.?\d*", r["response"].replace(",", "")) else 0.0
        for r in parse_responses(completions)
    ]


def format_reasoning_reward(completions, **kwargs) -> list[float]:
    return [0.5 if r["thinking_content"] and r["response"] else 0.0 for r in parse_responses(completions)]


GRPO_REWARD_FUNCS = [accuracy_reward, format_number_reward, format_reasoning_reward]

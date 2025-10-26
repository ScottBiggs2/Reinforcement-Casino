import re

def extract_boxed_answer(text):
    match = re.search(r"\\boxed\{(.*)\}", text)
    return match.group(1).strip() if match else None

def math_reward(prediction: str, target: str) -> int:
    pred_ans = extract_boxed_answer(prediction)
    gold_ans = extract_boxed_answer(target)
    return int(pred_ans == gold_ans)

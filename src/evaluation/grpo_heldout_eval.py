"""Held-out math evaluation for the matched-schedule GRPO arms.

Measurement is deliberately IDENTICAL to training: same prompt suffix
(LLAMA_COT_PROMPT_SUFFIX), same answer extraction (`accuracy_reward` from
src.utils.grpo_rewards). Only the data changes. Using lm_eval's GSM8K task instead
would introduce a prompt/extraction mismatch against a model GRPO-trained to emit
\\boxed{...}, so the score would partly measure template agreement rather than math.

GSM8K test is the benchmark because it cannot be contaminated: training consumed
~4000 shuffled OpenR1-Math-220k prompts, so any OpenR1 subset carries ~4% expected
overlap and no OpenR1 "held-out" split is defensible without the exact index list.

Per-problem correctness is written out so arms can be compared PAIRED (same problems,
McNemar) rather than as independent binomials.

Usage:
  python -m src.evaluation.grpo_heldout_eval --model_path <ckpt> --tag oracle97.5 \
      --out_dir /scratch/.../heldout_eval [--limit N]
"""
import argparse, json, os, time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.grpo_rewards import LLAMA_COT_PROMPT_SUFFIX, get_grpo_reward_funcs

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(args.limit))
    print(f"[{args.tag}] GSM8K test: {len(ds)} problems", flush=True)

    tok_src = args.model_path if os.path.exists(os.path.join(args.model_path, "tokenizer.json")) else BASE_MODEL
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left padding is required for correct batched generation with a decoder-only model.
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    accuracy_reward = get_grpo_reward_funcs("llama_cot")[0]

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q + LLAMA_COT_PROMPT_SUFFIX}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in ds["question"]
    ]
    golds = list(ds["answer"])

    records, t0 = [], time.time()
    for i in range(0, len(prompts), args.batch_size):
        chunk = prompts[i : i + args.batch_size]
        gold_chunk = golds[i : i + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy -> deterministic pass@1
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        scores = accuracy_reward(texts, gold_chunk)
        for j, (t, g, s) in enumerate(zip(texts, gold_chunk, scores)):
            ntok = int((gen[j] != tok.pad_token_id).sum())
            records.append(
                {
                    "idx": i + j,
                    "correct": bool(s == 1.0),
                    "gen_tokens": ntok,
                    "truncated": ntok >= args.max_new_tokens,
                    "gold": g.split("####")[-1].strip(),
                }
            )
        done = len(records)
        acc = sum(r["correct"] for r in records) / done
        el = time.time() - t0
        print(f"[{args.tag}] {done}/{len(prompts)}  acc={acc:.4f}  {el/60:.1f}min", flush=True)

    n = len(records)
    k = sum(r["correct"] for r in records)
    acc = k / n
    se = (acc * (1 - acc) / n) ** 0.5
    trunc = sum(r["truncated"] for r in records) / n
    mlen = sum(r["gen_tokens"] for r in records) / n

    summary = {
        "tag": args.tag,
        "model_path": args.model_path,
        "benchmark": "gsm8k_test",
        "n": n,
        "correct": k,
        "accuracy": acc,
        "se": se,
        "truncated_frac": trunc,
        "mean_gen_tokens": mlen,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "scorer": "src.utils.grpo_rewards.accuracy_reward (llama_cot) — identical to training",
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.tag}] DONE acc={acc:.4f} +/- {se:.4f}  trunc={trunc:.3f}  "
          f"mean_tokens={mlen:.0f}  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

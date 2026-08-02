"""Build a GRPO-trainable subset of lingshu-medical-mllm/ReasonMed.

WHY THIS EXISTS
ReasonMed ships three columns — instruction / input / output — and **no answer
column**. `input` is empty on every row. It is an SFT triplet dataset, so it cannot
be fed to GRPO as-is: `grpo_rewards.accuracy_reward(completions, solution, ...)`
requires a `solution` column and would raise without one.

WHAT THIS SCRIPT DOES
Recovers a usable target from the CoT itself: `output` frequently ends with a
sentence of the form "the correct answer is <option text>", and `instruction`
always carries lettered options. We match the stated text against the option list
and keep only rows where it hits **exactly one** option.

Measured keep rate: **12.5 %** over a shuffled stream (20,000 kept from 160,151
scanned, 2026-07-31). Note the first 500 rows of the unshuffled split give 40 %,
which is NOT representative — the head of the dataset is markedly cleaner than the
body, so never size this from an offset-0 sample. 12.5 % of 1.11 M is still ~139 k
rows against the 4,000 a 500-step × 8-generation run consumes, so yield is not a
constraint either way.

TWO PROPERTIES THAT MUST BE DISCLOSED WHEREVER THIS RUN IS REPORTED
1. `solution` is **ReasonMed's own CoT conclusion**, not the gold label from the
   source benchmarks (MedQA / MedMCQA / PubMedQA). ReasonMed does not carry those.
   So the reward is "agree with the teacher CoT", which is distillation-flavoured,
   NOT verifiable-reward RL. Do not label the resulting run RLVR.
2. The 40 % that survive extraction are a **biased subset** — rows where the teacher
   stated a single clean answer that matched one option. Harder or more equivocal
   questions (one sampled row concluded no option was correct) drop out.

WHY THE ANSWER IS EMITTED AS A NUMBER, NOT A LETTER
The `llama_cot` reward profile is three functions summing to 2.0, and one of them,
`format_number_reward`, pays 0.5 only if the response contains a numeric token. A
letter answer ("B") scores 0 there **always**, which would silently change the
reward composition relative to the OpenR1-Math run and contaminate the very mask
comparison this dataset is being prepared for. Emitting the 1-based option index
keeps all three reward terms live and identical to the math run, so dataset is the
only variable between the two GRPO arms.

Usage:
  python scripts/prep_reasonmed_mcq.py --out /scratch/$USER/datasets/reasonmed_mcq \
      --n_keep 20000
"""
import argparse
import re
import unicodedata

OPTION_RE = re.compile(r"^\s*([A-E])[\.\)]\s*(.+?)\s*$", re.M)
ANSWER_RE = re.compile(r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*\"?([^\.\"\n]{1,80})", re.I)

# The trailing format is NOT cosmetic. `_split_cot_heuristic` splits reasoning from
# answer on "Final Answer:", "####" or \boxed{} only -- it does NOT recognise "the
# answer is". Without a recognised marker the thinking span parses as empty and
# `format_reasoning_reward` pays 0 on every rollout, capping the medical arm at 1.5
# while the math arm reaches 2.0. Measured on the math dense run (last 50 steps):
# accuracy 0.0925, format_number 0.4775, format_reasoning 0.4625 -- the format terms
# sit near their 0.5 ceiling there, so the medical arm has to be able to reach them
# too or the two arms are optimising differently shaped objectives and the mask
# comparison is contaminated. Llama produces this marker reliably when asked.
SUFFIX = (
    "\n\nThink step by step, then end your reply with a line of exactly this form:\n"
    "Final Answer: <the number of the correct option>"
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower().strip()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def extract(rec):
    """-> {prompt, solution} or None if no unambiguous option match."""
    instruction = rec.get("instruction") or ""
    output = rec.get("output") or ""

    options = OPTION_RE.findall(instruction)
    if len(options) < 2:
        return None

    m = ANSWER_RE.search(output)
    if not m:
        return None
    said = _norm(m.group(1))
    if not said:
        return None

    hits = []
    for idx, (letter, text) in enumerate(options, start=1):
        t = _norm(text)
        if not t:
            continue
        # exact, or containment either way (the CoT often says "**Vitamin B**, with ...")
        if t == said or t in said or said in t:
            hits.append(idx)

    if len(set(hits)) != 1:
        return None  # zero matches, or genuinely ambiguous between options

    return {"prompt": instruction.strip() + SUFFIX, "solution": str(hits[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset directory (parquet)")
    ap.add_argument("--n_keep", type=int, default=20000)
    ap.add_argument("--hf_id", default="lingshu-medical-mllm/ReasonMed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import Dataset, load_dataset

    # Streaming: 1.11 M rows and we need ~20 k, so never materialise the whole thing.
    stream = load_dataset(args.hf_id, split="train", streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=10000)

    kept, seen = [], 0
    for rec in stream:
        seen += 1
        got = extract(rec)
        if got:
            kept.append(got)
            if len(kept) >= args.n_keep:
                break
        if seen % 20000 == 0:
            print(f"  scanned {seen:,}  kept {len(kept):,}  ({len(kept)/seen*100:.1f} %)")

    print(f"scanned {seen:,} rows, kept {len(kept):,} ({len(kept)/seen*100:.1f} %)")
    if not kept:
        raise SystemExit("no rows survived extraction — check the regexes against a sample")

    # Sanity: the answer index must be inside the option range, and the distribution
    # must not be degenerate (a constant target would train the model to say "1").
    from collections import Counter
    dist = Counter(r["solution"] for r in kept)
    print("solution distribution:", dict(sorted(dist.items())))
    top = dist.most_common(1)[0][1] / len(kept)
    if top > 0.60:
        print(f"WARNING: {top:.0%} of targets are a single option — check for extraction bias")

    ds = Dataset.from_list(kept)
    ds.to_parquet(f"{args.out}/data.parquet")
    print(f"wrote {len(ds)} rows -> {args.out}/data.parquet")
    print("columns:", ds.column_names)
    print("\nexample prompt:\n", ds[0]["prompt"][:400])
    print("\nexample solution:", ds[0]["solution"])


if __name__ == "__main__":
    main()

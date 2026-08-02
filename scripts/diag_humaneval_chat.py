#!/usr/bin/env python
"""Diagnose humaneval-under-chat-template: dump raw model output + every
filtered variant for a handful of docs, replicating coding_evaluator's task
modifications exactly. Run inside an eval job (1 GPU)."""

import functools
import json
import os
import re
import sys

sys.path.insert(0, "/work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed/src")

from evaluation.coding_evaluator import markdown_code_filter  # noqa: E402
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.filters.custom import CustomFilter  # noqa: E402
from lm_eval.tasks import get_task_dict  # noqa: E402

OUT = os.environ.get("DIAG_OUT", "/scratch/xie_yiyi_neu/rebuttal_analysis/eval_tulu3/diag_humaneval")
os.makedirs(OUT, exist_ok=True)

INSTRUCTION = "Provide ONLY the Python code block starting with 'def' and ending with the completed function logic. Do not include any explanation or markdown formatting outside the code block."


def _to_continuation(code, doc):
    entry = doc.get("entry_point", "")
    m = re.search(rf"^def\s+{re.escape(entry)}\s*\(.*?:\s*$", code, flags=re.M)
    if not m:
        return code
    pre = "".join(f"    {ln}\n" for ln in code[: m.start()].splitlines() if ln.strip())
    return "\n" + pre + code[m.end():]


def _humaneval_chat_filter(resps, docs):
    cleaned = [markdown_code_filter(list(r)) for r in resps]
    return [[_to_continuation(c, d) for c in resp] for resp, d in zip(cleaned, docs)]


def main():
    t_dict = get_task_dict(["humaneval"])
    task = t_dict["humaneval"]
    task.config.doc_to_text = f"{INSTRUCTION}\n\n{task.config.doc_to_text}"
    for ens in task._filters:
        ens.filters.insert(0, functools.partial(CustomFilter, filter_fn=_humaneval_chat_filter))
    gk = getattr(task.config, "generation_kwargs", None)
    print("generation_kwargs BEFORE clear:", gk)
    if isinstance(gk, dict):
        gk["until"] = []
    print("generation_kwargs AFTER clear:", task.config.generation_kwargs)

    results = simple_evaluate(
        model="vllm",
        model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=float16,max_model_len=4096,enable_chunked_prefill=False,max_num_batched_tokens=4096,max_num_seqs=32,gpu_memory_utilization=0.7,trust_remote_code=True",
        tasks=list(t_dict.values()),
        num_fewshot=0,
        limit=8,
        batch_size="auto",
        apply_chat_template=True,
        gen_kwargs={"temperature": 0.0, "max_gen_toks": 1024, "do_sample": False},
        log_samples=True,
        confirm_run_unsafe_code=True,
    )

    print("SCORE:", results["results"].get("humaneval"))
    samples = results.get("samples", {}).get("humaneval", [])
    dump = []
    for s in samples:
        dump.append({
            "doc_id": s.get("doc_id"),
            "entry_point": s.get("doc", {}).get("entry_point"),
            "raw_resps": s.get("resps"),
            "filtered_resps": s.get("filtered_resps"),
            "metrics": {k: v for k, v in s.items() if k.startswith("pass")},
        })
    with open(f"{OUT}/samples.json", "w") as f:
        json.dump({"score": results["results"].get("humaneval"), "samples": dump}, f, indent=1)
    print(f"dumped {len(dump)} samples -> {OUT}/samples.json")


if __name__ == "__main__":
    main()

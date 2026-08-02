#!/usr/bin/env python
"""One-shot LOGIN-NODE prefetch of every dataset the Tulu3 eval suite needs.

Run on login.aicr.ai (network + HF token available) with the rl_casino_eval env
and the same HF_HOME/HF_DATASETS_CACHE the jobs use, so that eval jobs can run
with HF_*_OFFLINE=1. Rationale: 9 concurrent jobs downloading into the shared
/scratch hf_cache deadlocked on datasets' filelocks over autofs (round 3,
jobs 246641-49, all TIMEOUT).

    conda activate $W/envs/rl_casino_eval
    python scripts/aicr_prefetch_eval_datasets.py

Exits non-zero if any benchmark group ends up with no loadable task.
"""

import os
import socket
import sys
import traceback

for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "HF_EVALUATE_OFFLINE"):
    os.environ.pop(var, None)
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

# Login-node IPv6 routes to huggingface.co blackhole (SYN timeout, verified
# 2026-08-01); urllib3 walks all 8 v6 addresses at the OS connect timeout
# before reaching v4, so requests without an explicit timeout hang ~17 min
# (the `evaluate` metric-script HEAD has none). Prefer IPv4 outright.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first_getaddrinfo(*args, **kwargs):
    infos = _orig_getaddrinfo(*args, **kwargs)
    v4 = [ai for ai in infos if ai[0] == socket.AF_INET]
    return v4 or infos


socket.getaddrinfo = _ipv4_first_getaddrinfo

# Candidate lists mirror each evaluator's task_candidates (first-success order);
# prefetch every candidate that resolves so the in-job pick is cached whichever
# name wins.
GROUPS = {
    "mmlu": ["mmlu"],
    "math": ["hendrycks_math", "math", "minerva_math"],
    "gsm8k": ["gsm8k"],
    "coding": ["humaneval", "mbpp"],
    "ifeval": ["ifeval"],
    "squad": ["squad_completion", "squad", "squad_v2"],
    "gpqa": ["gpqa_diamond_zeroshot", "gpqa_diamond", "gpqa_diamond_n-shot", "gpqa"],
}


def materialize(task):
    """Force the task's dataset download + doc materialization."""
    for attr in ("eval_docs", "test_docs", "validation_docs"):
        try:
            docs = getattr(task, attr)
            docs = docs() if callable(docs) else docs
            n = sum(1 for _ in docs)
            return n
        except Exception:
            continue
    return -1


def main():
    from lm_eval.tasks import TaskManager

    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")

    failed_groups = []
    for group, candidates in GROUPS.items():
        ok = []
        for name in candidates:
            try:
                loaded = TaskManager().load([name])
                for tname, task in loaded["tasks"].items():
                    n = materialize(task)
                    print(f"[{group}] {name} -> {tname}: {n} docs")
                ok.append(name)
            except Exception as e:
                print(f"[{group}] {name}: FAILED ({type(e).__name__}: {e})")
        if not ok:
            failed_groups.append(group)
        print(f"[{group}] cached candidates: {ok or 'NONE'}")

    # squad_evaluator's primary path bypasses lm-eval entirely.
    try:
        from datasets import load_dataset
        for split in ("train", "validation"):
            ds = load_dataset("squad", split=split)
            print(f"[squad-direct] {split}: {len(ds)} rows")
        import evaluate
        evaluate.load("squad")
        print("[squad-direct] evaluate.load('squad') cached")
    except Exception:
        traceback.print_exc()
        failed_groups.append("squad-direct")

    # humaneval's lm-eval task module runs evaluate.load("code_eval") at import
    # time — cache the metric script so offline jobs don't reach for the network.
    try:
        import evaluate
        evaluate.load("code_eval")
        print("[coding] evaluate.load('code_eval') cached")
    except Exception:
        traceback.print_exc()
        failed_groups.append("code_eval-metric")

    if failed_groups:
        print(f"\nPREFETCH INCOMPLETE: {failed_groups}")
        sys.exit(1)
    print("\nPREFETCH COMPLETE — jobs can run with HF_*_OFFLINE=1")


if __name__ == "__main__":
    main()

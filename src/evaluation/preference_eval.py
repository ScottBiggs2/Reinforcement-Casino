"""Held-out preference evaluation: DPO implicit reward margin and accuracy.

WHY THIS EXISTS
---------------
Every DPO entrypoint in this repo sets ``eval_dataset = None`` (DPO_train.py,
DPO_timing_baseline.py, sparse_DPO_timing_baseline.py, sparse_dpo_efficiency.py).
The only ``rewards/margins`` and ``rewards/accuracies`` numbers we have are TRL's
TRAINING-batch logs. For Light-R1 that is 500 steps x eff. batch 128 over ~3,060
pairs = ~21 epochs, which is why preference accuracy reads 1.0. Quoting a training
metric measured after 21 passes as the discriminative evidence in a rebuttal is not
defensible.

This script computes the same quantity on data the model was not trained on:

    r(x, y)  = beta * ( log pi_theta(y|x) - log pi_ref(y|x) )      [DPO implicit reward]
    margin   = r(x, y_chosen) - r(x, y_rejected)
    accuracy = mean( margin > 0 )

with a bootstrap CI over examples, so "oracle vs random" can be read as separated or
not separated rather than eyeballed.

CONTAMINATION
-------------
``src/utils/data_utils.load_dpo_dataset`` selects a subset as a deterministic PREFIX
(``select(range(n))``), never a shuffle. Consequences:

  * A run trained with ``--subset_size N`` leaves examples [N:] genuinely unseen, so
    ``--holdout_tail`` is a clean holdout for it.
  * A run trained with NO subset_size consumed the whole split. The Light-R1 runs in
    the submission are in this category: there is no in-domain holdout for them, and
    no flag to this script can manufacture one.

So this script does not guess. Pass ``--trained_on`` (or let it read run_manifest.json)
and it classifies the evaluation as CLEAN / CONTAMINATED / CROSS-DATASET and prints
that verdict next to every number. For the existing Light-R1 checkpoints the honest
configuration is cross-dataset (evaluate on a Tulu3 tail), reported as such.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# log-probability of a completion given a prompt
# ---------------------------------------------------------------------------

@torch.no_grad()
def sequence_logprob(model, prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
    """Sum of token log-probs of `completion_ids` conditioned on `prompt_ids`.

    Both are [B, T] with right padding; padding is assumed to be token id 0 and is
    excluded via an explicit length mask rather than by trusting the pad token.
    Returns [B].
    """
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)

    seq = torch.cat([prompt_ids, completion_ids], dim=1)
    attn = (seq != 0).long()
    # first completion position differs per row, so build the loss mask from lengths
    p_len = (prompt_ids != 0).sum(dim=1)
    c_len = (completion_ids != 0).sum(dim=1)

    logits = model(input_ids=seq, attention_mask=attn).logits[:, :-1]
    targets = seq[:, 1:]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = torch.gather(logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    pos = torch.arange(tok_lp.size(1), device=device).unsqueeze(0)
    # target index t corresponds to seq position t+1; completion occupies
    # [p_len, p_len + c_len), so in target space that is [p_len - 1, p_len + c_len - 1)
    lo = (p_len - 1).unsqueeze(1)
    hi = (p_len + c_len - 1).unsqueeze(1)
    mask = ((pos >= lo) & (pos < hi)).float()
    return (tok_lp * mask).sum(dim=1)


def encode_batch(tokenizer, texts: List[str], max_len: int) -> torch.Tensor:
    enc = [tokenizer(t, truncation=True, max_length=max_len, return_tensors="pt") for t in texts]
    padded = tokenizer.pad(enc, padding=True, return_tensors="pt")
    return padded["input_ids"].to(torch.long)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_policy(model_path: str, adapter: Optional[str], device_map: Optional[str]):
    """Load the policy. `adapter` attaches a PEFT adapter to `model_path` (the base)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device_map or "cuda",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        # merge so the forward path (and therefore timing) matches a plain model
        model = model.merge_and_unload()
        print(f"Attached and merged LoRA adapter: {adapter}")
    model.config.use_cache = False
    return model.eval()


# ---------------------------------------------------------------------------
# contamination classification
# ---------------------------------------------------------------------------

def classify(eval_key: str, trained_on: Optional[str], trained_subset: Optional[int],
             holdout_tail: Optional[int], n_total: int) -> Dict[str, object]:
    if trained_on is None:
        return {"verdict": "UNKNOWN",
                "detail": "no --trained_on given and no run_manifest.json found; "
                          "cannot certify the eval slice was unseen"}
    if trained_on != eval_key:
        return {"verdict": "CROSS-DATASET",
                "detail": f"trained on '{trained_on}', evaluated on '{eval_key}' — "
                          f"unseen by construction, but out of distribution"}
    if holdout_tail is None:
        return {"verdict": "CONTAMINATED",
                "detail": f"same dataset '{eval_key}' and no --holdout_tail"}
    if trained_subset is None:
        return {"verdict": "CONTAMINATED",
                "detail": f"the run consumed ALL of '{eval_key}' (no subset_size), so the "
                          f"tail was seen too. data_utils selects a prefix, so a holdout "
                          f"only exists if the run used --subset_size."}
    if trained_subset <= n_total - holdout_tail:
        return {"verdict": "CLEAN",
                "detail": f"run saw prefix [0:{trained_subset}); eval slice is the last "
                          f"{holdout_tail} of {n_total} — disjoint"}
    return {"verdict": "CONTAMINATED",
            "detail": f"run saw prefix [0:{trained_subset}) which overlaps the last "
                      f"{holdout_tail} of {n_total}"}


def bootstrap_ci(values: List[float], n_boot: int, seed: int, alpha: float = 0.05):
    t = torch.tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    n = t.numel()
    means = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        means[b] = t[idx].mean()
    lo = torch.quantile(means, alpha / 2).item()
    hi = torch.quantile(means, 1 - alpha / 2).item()
    return lo, hi


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out DPO preference evaluation")
    ap.add_argument("--model", required=True, help="Policy: HF id or checkpoint dir.")
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (policy = --model + adapter).")
    ap.add_argument("--ref_model", required=True, help="Reference: the pretrained base (theta^0).")
    ap.add_argument("--eval_dataset", required=True, help="Registry key, e.g. light-r1 / tulu3.")
    ap.add_argument("--holdout_tail", type=int, default=None,
                    help="Evaluate on the LAST N examples. Disjoint from any run that used "
                         "--subset_size <= len-N, since subsetting takes a prefix.")
    ap.add_argument("--n_eval", type=int, default=500, help="Cap on evaluated pairs.")
    ap.add_argument("--trained_on", default=None, help="Dataset key the policy was trained on.")
    ap.add_argument("--trained_subset", type=int, default=None,
                    help="subset_size the policy was trained with (None = whole split).")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta (must match training).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Write JSON here (default: alongside --model).")
    ap.add_argument("--tag", default=None, help="Label for this condition in the output.")
    args = ap.parse_args()

    # let the run's own manifest fill in provenance when the caller did not
    if args.trained_on is None:
        for cand in (os.path.join(args.model, "run_manifest.json"),
                     os.path.join(os.path.dirname(args.model.rstrip("/")), "run_manifest.json")):
            if os.path.isfile(cand):
                try:
                    man = json.load(open(cand))
                    args.trained_on = args.trained_on or man.get("dataset_key")
                    if args.trained_subset is None:
                        args.trained_subset = man.get("subset_size")
                    print(f"Provenance from {cand}: trained_on={args.trained_on}, "
                          f"subset={args.trained_subset}")
                except Exception as exc:
                    print(f"could not read {cand}: {exc}")
                break

    from src.utils.dataset_registry import load_dpo_dataset

    ds = load_dpo_dataset(args.eval_dataset)
    n_total = len(ds)
    if args.holdout_tail:
        start = max(0, n_total - args.holdout_tail)
        ds = ds.select(range(start, n_total))
        print(f"Holdout slice: [{start}:{n_total}] of {n_total}")
    if args.n_eval and len(ds) > args.n_eval:
        ds = ds.select(range(args.n_eval))

    status = classify(args.eval_dataset, args.trained_on, args.trained_subset,
                      args.holdout_tail, n_total)
    print("\n" + "=" * 66)
    print(f"CONTAMINATION VERDICT: {status['verdict']}")
    print(f"  {status['detail']}")
    print("=" * 66 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Two passes, one model resident at a time. Holding policy AND reference together
    # is ~32 GB for two 8B models in bf16, which restricts the job to the single H200
    # node in this cluster's multigpu partition and puts it behind a multi-hour queue.
    # Sequential passes halve peak memory and let the eval run on an A100, off the
    # critical path of the training runs that genuinely need H200.
    def collect(model) -> Dict[str, List[float]]:
        lp_c: List[float] = []
        lp_r: List[float] = []
        for i in range(0, len(ds), args.batch_size):
            rows = ds[i: i + args.batch_size]
            p = encode_batch(tokenizer, list(rows["prompt"]), args.max_prompt_length)
            c = encode_batch(tokenizer, list(rows["chosen"]), args.max_length)
            r = encode_batch(tokenizer, list(rows["rejected"]), args.max_length)
            lp_c += sequence_logprob(model, p, c).tolist()
            lp_r += sequence_logprob(model, p, r).tolist()
            if (i // args.batch_size) % 25 == 0:
                print(f"    {min(i + args.batch_size, len(ds)):>5}/{len(ds)} pairs", flush=True)
        return {"chosen": lp_c, "rejected": lp_r}

    print("\nPass 1/2 — policy")
    policy = load_policy(args.model, args.adapter, args.device_map)
    pol = collect(policy)
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nPass 2/2 — reference")
    ref = load_policy(args.ref_model, None, args.device_map)
    rf = collect(ref)
    del ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    r_chosen_all = [args.beta * (a - b) for a, b in zip(pol["chosen"], rf["chosen"])]
    r_rejected_all = [args.beta * (a - b) for a, b in zip(pol["rejected"], rf["rejected"])]
    margins = [a - b for a, b in zip(r_chosen_all, r_rejected_all)]

    n = len(margins)
    acc = [1.0 if m > 0 else 0.0 for m in margins]
    m_lo, m_hi = bootstrap_ci(margins, args.n_bootstrap, args.seed)
    a_lo, a_hi = bootstrap_ci(acc, args.n_bootstrap, args.seed)

    result = {
        "tag": args.tag or os.path.basename(args.model.rstrip("/")),
        "model": args.model,
        "adapter": args.adapter,
        "ref_model": args.ref_model,
        "eval_dataset": args.eval_dataset,
        "holdout_tail": args.holdout_tail,
        "n_pairs": n,
        "beta": args.beta,
        "contamination": status,
        "reward_margin_mean": sum(margins) / n,
        "reward_margin_ci95": [m_lo, m_hi],
        "preference_accuracy": sum(acc) / n,
        "preference_accuracy_ci95": [a_lo, a_hi],
        # Report both implicit rewards, not just the margin: at an aggressive LR, DPO
        # can drive chosen and rejected DOWN together while the margin still grows
        # (likelihood displacement, Razin et al. ICLR 2025). A margin alone hides that.
        "reward_chosen_mean": sum(r_chosen_all) / n,
        "reward_rejected_mean": sum(r_rejected_all) / n,
    }

    out = args.out or os.path.join(args.model, "preference_eval.json")
    try:
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nWrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}")

    print("\n" + "=" * 66)
    for k in ("tag", "eval_dataset", "n_pairs", "reward_margin_mean", "reward_margin_ci95",
              "preference_accuracy", "preference_accuracy_ci95",
              "reward_chosen_mean", "reward_rejected_mean"):
        print(f"  {k}: {result[k]}")
    print(f"  contamination: {status['verdict']}")
    print("=" * 66)


if __name__ == "__main__":
    main()

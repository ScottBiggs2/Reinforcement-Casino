#!/usr/bin/env bash
# Runs ON a Prime Intellect H100 box. Single-variable test of the Appendix D.4 confound.
#
# THE QUESTION
# D.4 reports that at rho=97.5% the GRPO oracle mask is indistinguishable from random
# and far below dense. Reading training_args.bin out of those three checkpoints shows
# the arms were not comparable:
#
#            scheduler   max_steps   LR at step 500   % of peak
#   dense    COSINE      5000        4.990e-06        99.8%
#   sparse   LINEAR      500         1.111e-08         0.2%
#
# a 449x gap, with the two arms' learning rates moving in OPPOSITE directions across
# the comparison window (dense was still inside its 500-step warmup ramp; sparse had
# annealed to zero).
#
# THE TEST
# Rerun dense and sparse with an identical schedule, changing nothing else. Completion
# cap stays at the as-run 1024 -- raising it too would make this a two-variable test and
# the diagnosis would be unreadable. 150 steps, because both outcomes are writable:
#   sparse tracks dense   -> D.4's gap was the schedule; the GRPO conclusion is withdrawn
#   sparse still lags     -> D.4's conclusion may survive; we say so and report both
#
# NOT measured here: wall-clock or throughput. Those must stay on Discovery's H200 to
# remain comparable with Tables 2/3.
set -euo pipefail

STEPS="${STEPS:-150}"
CAP="${CAP:-1024}"
ROOT="${ROOT:-/workspace}"
REPO="$ROOT/Reinforcement-Casino"
DATA="$ROOT/data"
OUT="$ROOT/out"
MASK="$DATA/oracle_grpo_math220k_step500_sp97.5.pt"

export HF_HOME="$DATA/hf_cache"
export HF_DATASETS_CACHE="$DATA/hf_cache/datasets"
export HF_HUB_OFFLINE=1          # everything was pushed from Discovery; never call the hub
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_MODE=disabled WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
cd "$REPO"

banner () { echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

banner "Preflight"
python - <<'PY'
import torch, transformers, trl
print("torch       ", torch.__version__, "| cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu          {p.name}  {p.total_memory/1e9:.0f} GB")
PY
test -f "$MASK" || { echo "FATAL: mask not found at $MASK"; exit 1; }
echo "mask: $(du -h "$MASK" | cut -f1)"

# ---------------------------------------------------------------- dense arm
banner "ARM 1/2  dense GRPO  |  ${STEPS} steps  |  cosine  |  cap ${CAP}"
python src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --num_steps "$STEPS" \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --beta 0.025 \
    --max_grad_norm 0.1 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_prompt_length 512 \
    --max_completion_length "$CAP" \
    --save_steps 100000 \
    --output_base_dir "$OUT/dense" \
    --run_name "matched_dense_${STEPS}steps_cap${CAP}" 2>&1 | tee "$OUT/dense.log"

# --------------------------------------------------------------- sparse arm
# --lr_scheduler_type is NEW on this branch. sparse_grpo_bsr.py previously had no
# scheduler argument at all, so it silently took the HF default (linear) while
# GRPO_train.py defaulted to cosine -- that omission IS the D.4 confound.
banner "ARM 2/2  sparse GRPO oracle @97.5%  |  ${STEPS} steps  |  cosine  |  cap ${CAP}"
python src/full_training/sparse_grpo_bsr.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --mask "$MASK" \
    --dataset math-220k \
    --n_steps "$STEPS" \
    --lr 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps $(python -c "print(int(0.1*$STEPS))") \
    --grpo_beta 0.025 \
    --max_grad_norm 0.1 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_prompt_length 512 \
    --max_completion_length "$CAP" \
    --optimizer sparse_adamw \
    --save_steps 100000 \
    --save_model false \
    --output_base_dir "$OUT/sparse" \
    --run_name "matched_sparse_oracle_${STEPS}steps_cap${CAP}" 2>&1 | tee "$OUT/sparse.log"

banner "Comparison"
python - "$OUT" <<'PY'
import json, re, sys, statistics as st, pathlib
out = pathlib.Path(sys.argv[1])

def parse(logpath):
    """Trainer prints one dict per step; pull the metrics we need."""
    rows = []
    for line in open(logpath, errors="ignore"):
        line = line.strip()
        if not (line.startswith("{") and "'reward'" in line):
            continue
        try:
            rows.append(eval(line, {"__builtins__": {}}, {"nan": float("nan"), "inf": float("inf")}))
        except Exception:
            pass
    return rows

def summarize(rows, label):
    if not rows:
        print(f"{label:<10} no parsed steps"); return None
    g = lambda k: [r[k] for r in rows if k in r and r[k] is not None]
    rew, lr = g("reward"), g("learning_rate")
    acc = g("rewards/accuracy_reward/mean")
    clip = g("completions/clipped_ratio")
    asd, rsd = g("rewards/accuracy_reward/std"), g("reward_std")
    share = [a / r for a, r in zip(asd, rsd) if r > 1e-9]
    n = len(rew)
    first, last = rew[: max(1, n // 5)], rew[-max(1, n // 5):]
    print(f"{label:<10}{n:>7}{st.mean(first):>12.4f}{st.mean(last):>12.4f}"
          f"{(st.mean(last)-st.mean(first)):>12.4f}{st.mean(acc or [0]):>10.4f}"
          f"{st.mean(clip or [0]):>10.3f}{(st.mean(share) if share else float('nan')):>12.3f}"
          f"{(lr[-1] if lr else float('nan')):>12.3e}")
    return {"n": n, "reward_first20pct": st.mean(first), "reward_last20pct": st.mean(last),
            "delta": st.mean(last) - st.mean(first), "acc_mean": st.mean(acc or [0]),
            "clipped": st.mean(clip or [0]),
            "acc_signal_share": st.mean(share) if share else None,
            "final_lr": lr[-1] if lr else None}

print(f"{'arm':<10}{'steps':>7}{'rew_first':>12}{'rew_last':>12}{'delta':>12}"
      f"{'acc':>10}{'clip':>10}{'acc_sig':>12}{'final_lr':>12}")
res = {}
for name in ("dense", "sparse"):
    res[name] = summarize(parse(out / f"{name}.log"), name)

print()
print("Reference — the ORIGINAL runs at 500 steps with MISMATCHED schedules:")
print("  dense   acc 0.0779  clip 0.858  acc_sig 0.415  final_lr 4.990e-06")
print("  sparse  acc 0.0535  clip 0.912  acc_sig 0.237  final_lr 1.111e-08")
print()
if res.get("dense") and res.get("sparse"):
    d, s = res["dense"], res["sparse"]
    if d["final_lr"] and s["final_lr"]:
        ratio = d["final_lr"] / s["final_lr"] if s["final_lr"] else float("inf")
        print(f"final-LR ratio dense/sparse = {ratio:.2f}x   (was 449x in the submitted runs)")
        print("  -> schedules are matched" if 0.5 < ratio < 2 else "  -> STILL MISMATCHED, do not interpret")
    print(f"reward gain  dense {d['delta']:+.4f}   sparse {s['delta']:+.4f}")
json.dump(res, open(out / "comparison.json", "w"), indent=2)
print(f"\nwrote {out/'comparison.json'}")
PY

banner "Done — REMEMBER TO DESTROY THE INSTANCE (billed hourly)"

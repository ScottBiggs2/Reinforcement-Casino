#!/bin/bash
# Read-only watchdog for the NeurIPS #29841 rebuttal queue. Takes no action -- it prints
# a status block and, when a watched run finishes, evaluates its acceptance gate.
#
# Deliberately does NOT scancel, resubmit, or edit anything. Every failure mode in this
# campaign so far has needed judgement (a failed LR gate means the config is wrong, so a
# blind resubmit just burns another 3 h).
#
# Usage:  bash scripts/rc_watch.sh
set -uo pipefail

# Runs on either cluster: AICR (primary since 2026-07-29) or Explorer (fallback queue).
# Detected from $USER so the same file works over `ssh aicr` and `ssh Discovery_Cluster`.
if [ "${USER}" = "xie_yiyi_neu" ]; then
  CLUSTER="AICR (primary)"
  PY="${PY:-/work/neu/p2026_0038_neu/xie_yiyi/envs/rl_casino/bin/python}"
  S="${S:-/scratch/xie_yiyi_neu}"
else
  CLUSTER="Explorer (fallback)"
  PY="${PY:-/home/xie.yiyi/.conda/envs/rl_casino/bin/python}"
  S="${S:-/scratch/xie.yiyi}"
fi
SINCE="${SINCE:-2026-07-26}"

echo "===== RC WATCH  $CLUSTER  $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

echo
echo "--- QUEUE ---"
q=$(squeue -u "$USER" -o "%.12i %.13P %.24j %.2t %.11M %.11L %R" 2>/dev/null)
echo "$q"
n_run=$(squeue -u "$USER" -h -t RUNNING 2>/dev/null | wc -l)
n_pd=$(squeue -u "$USER" -h -t PENDING 2>/dev/null | wc -l)
echo "running=$n_run pending=$n_pd"

echo
echo "--- TERMINAL STATES since $SINCE (anything not COMPLETED is worth reading) ---"
sacct -X -u "$USER" -S "$SINCE" \
      -o JobID%14,JobName%24,State%14,Elapsed,End%17,MaxRSS,NodeList%8 2>/dev/null \
  | grep -vE "PENDING|RUNNING" | tail -25

echo
echo "--- BROKEN DEPENDENCY CHECK ---"
# afterok chains die silently: the parent fails, children sit in Dependency forever.
for j in $(squeue -u "$USER" -h -o "%i" 2>/dev/null); do
  dep=$(scontrol show job "$j" 2>/dev/null | grep -o "Dependency=[^ ]*")
  case "$dep" in
    *unfulfilled*)
      parent=$(echo "$dep" | grep -oE "[0-9]+" | head -1)
      pstate=$(sacct -X -j "$parent" -o State -n 2>/dev/null | head -1 | tr -d ' ')
      if [ -n "$pstate" ] && [ "$pstate" != "PENDING" ] && [ "$pstate" != "RUNNING" ] \
         && [ "$pstate" != "COMPLETED" ]; then
        echo "  !! $j waits on $parent which is $pstate -- CHAIN IS DEAD"
      fi
      ;;
  esac
done
echo "  (no output above = no chain broken)"

echo
echo "--- ACCEPTANCE GATES ---"

# Gate 1: random-mask GRPO control must reproduce the 2026-04 schedule exactly.
CTRL=$(find "$S/rebuttal_analysis/grpo_random_control" -name trainer_state.json 2>/dev/null | sort | tail -1)
if [ -n "$CTRL" ]; then
  echo "  [8769965 random control] $CTRL"
  "$PY" - "$CTRL" <<'EOF'
import json, sys, statistics
d = json.load(open(sys.argv[1]))
h = d.get("log_history", [])
lr = [l["learning_rate"] for l in h if "learning_rate" in l]
K = "rewards/accuracy_reward/mean"
a = [l[K] for l in h if K in l]
print(f"    step={d.get('global_step')}  final_lr={lr[-1] if lr else None}")
if lr:
    ok = abs(lr[-1] - 1.1111111111111112e-08) / 1.1111111111111112e-08 < 0.05
    print(f"    GATE final_lr==1.111e-08 : {'PASS' if ok else 'FAIL -- schedule did not replicate, NOT a valid control'}")
if len(a) >= 100:
    f = statistics.mean(a[:50]); l_ = statistics.mean(a[-50:])
    print(f"    acc_reward first50={f:.4f} last50={l_:.4f} delta={l_-f:+.4f}")
    print(f"    compare to: in-task 0.0434->0.0833 | DPO-LightR1 0.0587->0.0809 | DPO-Tulu3 0.0561->0.0956")
EOF
else
  echo "  [8769965 random control] no trainer_state yet"
fi

# Gate 2: matched sparse GRPO must have picked up the max_grad_norm fix (deployed 07-27).
SP=$(find "$S/rebuttal_analysis/grpo_matched/sparse" -name training_args.bin 2>/dev/null | sort | tail -1)
if [ -n "$SP" ]; then
  echo "  [8734159_1 matched sparse] $SP"
  "$PY" - "$SP" <<'EOF'
import torch, sys
a = torch.load(sys.argv[1], weights_only=False)
mgn = getattr(a, "max_grad_norm", None)
sch = getattr(a, "lr_scheduler_type", None)
print(f"    max_grad_norm={mgn}  scheduler={sch}  max_steps={getattr(a,'max_steps',None)}")
print(f"    GATE max_grad_norm==0.1 : {'PASS' if mgn == 0.1 else 'FAIL -- clipping fix did not take, arm is NOT matched to dense'}")
EOF
else
  echo "  [8734159_1 matched sparse] no checkpoint yet"
fi

echo
echo "===== END ====="

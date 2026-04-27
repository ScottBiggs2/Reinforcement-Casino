#!/bin/bash
# Cluster-side persistent monitor for Phase 1 dense runs.
# Run with: nohup bash overnight_monitor_phase1.sh > /dev/null 2>&1 &
# Output goes to: $LOG (defined below).

JOBS="6357774 6357775 6357776"
LOG=/home/xie.yiyi/rc-sparse-speed/logs/overnight_monitor_phase1.log

declare -A LAST
declare -A REQ
declare -A WURL
declare -A STARTED
for J in $JOBS; do LAST[$J]=""; REQ[$J]=0; WURL[$J]=""; STARTED[$J]=0; done

{
echo "===== monitor started at $(date) ====="
echo "watching jobs: $JOBS"
echo ""

while true; do
  ALL_DONE=1
  for J in $JOBS; do
    SLINE=$(squeue -j $J -h -o "%T" 2>/dev/null)
    if [ -z "$SLINE" ]; then
      if [ "${LAST[$J]}" != "DONE" ]; then
        SACCT=$(sacct -j $J --format=JobID,JobName,State,Elapsed,ExitCode --noheader 2>/dev/null | head -1 | tr -s " ")
        echo "[$(date +%Y-%m-%d_%H:%M:%S)] === JOB $J FINISHED ==="
        echo "  sacct: $SACCT"
        echo "  requeues during run: ${REQ[$J]}"
        echo "  wandb URL: ${WURL[$J]:-not_captured}"
        OUTLOG=$(ls /home/xie.yiyi/rc-sparse-speed/logs/transfer_p1_*${J}.out 2>/dev/null | head -1)
        if [ -n "$OUTLOG" ]; then
          echo "  --- last 40 lines of stdout ---"
          tail -40 "$OUTLOG" 2>&1 | sed "s/^/    /"
          echo "  --- end ---"
        fi
        # Capture latest ckpt + summary from trainer_state.json
        for D in /scratch/xie.yiyi/transfer_v1/dense_*_llama8b; do
          LATEST_CKPT=$(find "$D" -maxdepth 4 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
          [ -z "$LATEST_CKPT" ] && continue
          # Heuristic: match this job by checking if .out log mentions this output dir
          if [ -n "$OUTLOG" ] && grep -q "$D" "$OUTLOG" 2>/dev/null; then
            TS="$LATEST_CKPT/trainer_state.json"
            if [ -f "$TS" ]; then
              echo "  --- final trainer_state summary (from $LATEST_CKPT) ---"
              /home/xie.yiyi/.conda/envs/rl_casino/bin/python -c "
import json, statistics
with open('$TS') as f: ts = json.load(f)
hist = ts.get('log_history', [])
print(f'    global_step: {ts.get(\"global_step\")}, max_steps: {ts.get(\"max_steps\")}, epoch: {ts.get(\"epoch\")}')
keys_dpo = ['loss', 'rewards/margins', 'rewards/accuracies', 'rewards/chosen', 'rewards/rejected', 'grad_norm']
keys_grpo = ['reward', 'rewards/accuracy_reward/mean', 'rewards/format_reasoning_reward/mean', 'kl', 'grad_norm']
for k in (keys_dpo + keys_grpo):
    vals = [h[k] for h in hist if k in h and h[k] is not None]
    if vals:
        print(f'    {k:46s}  first={vals[0]:.4f}  last={vals[-1]:.4f}  min={min(vals):.4f}  max={max(vals):.4f}')
" 2>&1 | sed "s/^/    /"
              echo "  --- end summary ---"
            fi
          fi
        done
        echo ""
        LAST[$J]="DONE"
      fi
    else
      ALL_DONE=0
      if [ "$SLINE" != "${LAST[$J]}" ]; then
        echo "[$(date +%Y-%m-%d_%H:%M:%S)] $J state: ${LAST[$J]:-init} -> $SLINE"
        if [ "${LAST[$J]}" = "RUNNING" ] && [ "$SLINE" = "PENDING" ]; then
          REQ[$J]=$((${REQ[$J]} + 1))
          echo "  ⚠️ requeue #${REQ[$J]} detected — auto-resume from latest ckpt next start"
        fi
        if [ "$SLINE" = "RUNNING" ] && [ "${STARTED[$J]}" = "0" ]; then
          echo "  🚀 job $J started running"
          STARTED[$J]=1
        fi
        LAST[$J]=$SLINE
      fi
      if [ "$SLINE" = "RUNNING" ] && [ -z "${WURL[$J]}" ]; then
        OUTLOG=$(ls /home/xie.yiyi/rc-sparse-speed/logs/transfer_p1_*${J}.out 2>/dev/null | head -1)
        if [ -n "$OUTLOG" ] && [ -s "$OUTLOG" ]; then
          URL=$(grep -hoE "https://wandb.ai/[a-zA-Z0-9_/?=&%-]+" "$OUTLOG" 2>/dev/null | head -1)
          if [ -n "$URL" ]; then
            echo "[$(date +%Y-%m-%d_%H:%M:%S)] $J wandb URL captured: $URL"
            WURL[$J]="$URL"
          fi
        fi
      fi
    fi
  done
  if [ $ALL_DONE -eq 1 ]; then
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] ===== ALL JOBS DONE — monitor exiting ====="
    break
  fi
  sleep 120
done
} >> "$LOG" 2>&1

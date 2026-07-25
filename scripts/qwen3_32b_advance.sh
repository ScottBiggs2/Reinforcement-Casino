#!/bin/bash
# Stateless, re-entrant pipeline advancer for the Qwen3-32B high-sparsity DPO chain.
#   p1 dense DPO  ->  p2 oracle mask @97.5%  ->  p3 sparse DPO
#
# Replaces the broken `--dependency=afterok` chain. The old chain never fired
# because p1 hits the 8h walltime (exit = TIME LIMIT, NOT 0), so afterok p2/p3
# stayed PENDING forever. This script instead inspects the filesystem and submits
# exactly the ONE job the pipeline needs next, then each training job calls this
# script again from its tail. Net effect: the chain self-advances across an
# arbitrary number of 8h slots with no human in the loop.
#
# Called two ways:
#   1) Kickoff (login node):   bash scripts/qwen3_32b_advance.sh
#   2) From a job's tail:       SLURM_JOB_ID is set -> excluded from the squeue guard
#
# Safety nets:
#   - squeue guard: never double-submits a phase that is already R/PD.
#   - progress guard: if a training phase makes NO checkpoint progress across 2
#     consecutive slots (resume silently broken / idle-killed during load), it
#     writes a STALL sentinel and halts instead of burning days of compute.
set -uo pipefail

REPO="/home/xie.yiyi/rc-sparse-speed"
cd "$REPO"
S="$REPO/scripts"

DATASET="${DATASET:-light-r1}"
DS_TAG=$(echo "$DATASET" | tr '-' '_')
TARGET=500
SPARSITY=97.5

DENSE_OUT="/scratch/xie.yiyi/transfer_v1/dense_dpo_${DS_TAG}_qwen3_32b"
ORACLE_DIR="/scratch/xie.yiyi/transfer_v1/oracle_masks_qwen3_32b"
MASK="$ORACLE_DIR/oracle_dpo_${DS_TAG}_step500_sp${SPARSITY}.pt"
RUN_TAG="oracle_step500"
SPARSE_OUT="/scratch/xie.yiyi/transfer_v1/sparse_dpo_${DS_TAG}_qwen3_32b_${RUN_TAG}"

STATE="/scratch/xie.yiyi/transfer_v1/pipeline_state.env"
STALL_SENTINEL="/scratch/xie.yiyi/transfer_v1/PIPELINE_STALLED"
DONE_SENTINEL="/scratch/xie.yiyi/transfer_v1/PIPELINE_DONE"
LOG="$REPO/docs/run_logs/qwen3_32b_autopipeline.log"
mkdir -p "$(dirname "$STATE")" "$(dirname "$LOG")"

SELF="${SLURM_JOB_ID:-none}"

log() { echo "[$(date '+%F %T')] [advance] $*" | tee -a "$LOG"; }

# Latest COMPLETE checkpoint step under a dir (requires trainer_state.json so a
# partially-written checkpoint is never counted / resumed from). Echoes 0 if none.
latest_step() {
  local base="$1" d best=0 s
  [ -d "$base" ] || { echo 0; return; }
  while IFS= read -r d; do
    [ -f "$d/trainer_state.json" ] || continue
    s=$(basename "$d" | sed 's/checkpoint-//')
    [[ "$s" =~ ^[0-9]+$ ]] || continue
    (( s > best )) && best=$s
  done < <(find "$base" -maxdepth 4 -type d -name 'checkpoint-*' 2>/dev/null)
  echo "$best"
}

# Is a job of this name already queued/running (excluding the calling job)?
phase_active() {
  local jobname="$1"
  squeue -u "$USER" -h -n "$jobname" -o '%i' 2>/dev/null \
    | grep -vx "$SELF" | grep -q .
}

state_get() { grep -E "^$1=" "$STATE" 2>/dev/null | tail -1 | cut -d= -f2-; }
state_set() {
  touch "$STATE"
  grep -vE "^$1=" "$STATE" > "$STATE.tmp" 2>/dev/null || true
  echo "$1=$2" >> "$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

if [ -f "$STALL_SENTINEL" ]; then
  log "STALL sentinel present ($STALL_SENTINEL); chain halted. Remove it after inspecting to resume."
  exit 1
fi

P1=$(latest_step "$DENSE_OUT")
P3=$(latest_step "$SPARSE_OUT")
[ -f "$MASK" ] && MASK_OK=1 || MASK_OK=0

# ---- decide the single next phase ----
if   [ "$P1" -lt "$TARGET" ]; then PHASE=p1
elif [ "$MASK_OK" -eq 0 ];    then PHASE=p2
elif [ "$P3" -lt "$TARGET" ]; then PHASE=p3
else PHASE=done
fi

log "state: p1_step=$P1 mask=$MASK_OK p3_step=$P3 -> next=$PHASE (self=$SELF)"

if [ "$PHASE" = "done" ]; then
  touch "$DONE_SENTINEL"
  log "PIPELINE COMPLETE: dense=$P1 mask=ok sparse=$P3. Nothing more to submit."
  exit 0
fi

case "$PHASE" in
  p1) JOBNAME="qwen3_32b_p1_dense"  ;;
  p2) JOBNAME="qwen3_32b_p2_oracle" ;;
  p3) JOBNAME="qwen3_32b_p3_sparse" ;;
esac

if phase_active "$JOBNAME"; then
  log "$PHASE ($JOBNAME) already queued/running; not submitting a duplicate."
  exit 0
fi

# ---- progress guard for training phases ----
if [ "$PHASE" = "p1" ] || [ "$PHASE" = "p3" ]; then
  [ "$PHASE" = "p1" ] && cur="$P1" || cur="$P3"
  prev=$(state_get "${PHASE}_last_step"); prev=${prev:--1}
  stall=$(state_get "${PHASE}_stall");    stall=${stall:-0}
  if [ "$prev" -ge 0 ] && [ "$cur" -le "$prev" ]; then
    stall=$((stall + 1))
    log "WARN: $PHASE made no checkpoint progress this slot (step stuck at $cur, prev=$prev), stall=$stall/2"
    if [ "$stall" -ge 2 ]; then
      state_set "${PHASE}_stall" "$stall"
      touch "$STALL_SENTINEL"
      log "STALL HALT: $PHASE stuck at step $cur over 2 consecutive slots. Resume is likely not working (or jobs are idle-killed during load). Chain stopped — inspect logs/${JOBNAME}_*.out, then remove $STALL_SENTINEL to retry."
      exit 1
    fi
  else
    stall=0
  fi
  state_set "${PHASE}_stall" "$stall"
  state_set "${PHASE}_last_step" "$cur"
fi

# ---- submit the next job ----
case "$PHASE" in
  p1)
    JID=$(sbatch --parsable --export=ALL,DATASET="$DATASET" "$S/qwen3_32b_p1_dense_dpo.sbatch")
    log "submitted p1 dense DPO (resume@step $P1) -> job $JID" ;;
  p2)
    JID=$(sbatch --parsable --export=ALL,DATASET="$DATASET" "$S/qwen3_32b_p2_oracle_dpo.sbatch")
    log "submitted p2 oracle mask (from dense checkpoint-$P1) -> job $JID" ;;
  p3)
    JID=$(sbatch --parsable --export=ALL,DATASET="$DATASET",MASK_PATH="$MASK",RUN_TAG="$RUN_TAG" "$S/qwen3_32b_p3_sparse_dpo.sbatch")
    log "submitted p3 sparse DPO (resume@step $P3, mask=$MASK) -> job $JID" ;;
esac

exit 0

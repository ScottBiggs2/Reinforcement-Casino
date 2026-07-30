#!/bin/bash
# Cluster-side autorunner for the oracle sparsity sweep.
#
# Runs AS A SLURM JOB on a CPU partition, so it survives laptop sleep, network loss and
# session end -- unlike a watcher living in an agent session (one of those died silently
# overnight on 2026-07-27 and nothing was noticed until morning).
#
# Design follows scripts/qwen3_32b_advance.sh, which is the validated pattern in this
# repo: stateless and re-entrant, inspect real state, submit exactly what is missing,
# guard against double submission.
#
# It deliberately does NOT blind-retry everything. scripts/rc_watch.sh was written
# read-only for a reason: most failures in this campaign need judgement, and a blind
# resubmit just burns another six hours. So failures are split in two:
#
#   AUTO-RESUBMIT (infrastructure; a retry is genuinely the right answer)
#     NODE_FAIL, PREEMPTED, CANCELLED-while-short-of-walltime (the RC idle-GPU killer),
#     and TIMEOUT *for sparse arms only*.
#
#   HALT AND REPORT (needs a human decision; retrying is wrong or wasteful)
#     - TIMEOUT on the DENSE job. DPO_train.py:308 skips the weight-delta callback on
#       resume, and deltas through step 500 are the entire point of that rerun. An
#       automatic resume would silently reproduce the exact failure this rebuild exists
#       to undo. Never auto-resume it.
#     - OUT_OF_MEMORY. A config problem. rho=80 is the untested arm (its int64 nonzero
#       index tensor is 12.9 GB on GPU vs 1.6 GB at rho=97.5); if it OOMs the request
#       needs changing, not repeating.
#     - Any FAILED whose .err does not match a known-fixable class.
#     - Any arm that has already been retried MAX_RETRY times.
#
# It never runs scancel and never deletes anything.
#
#SBATCH --job-name=sweep_autorunner
#SBATCH --output=logs/sweep_autorunner_%j.out
#SBATCH --error=logs/sweep_autorunner_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --requeue

set -uo pipefail

REPO="/home/xie.yiyi/rc-sparse-speed"
cd "$REPO" || exit 1
mkdir -p logs

STATE_DIR="/scratch/$USER/transfer_v1/autorunner"
mkdir -p "$STATE_DIR"
LOG="$REPO/docs/run_logs/sweep_autorunner.log"
HALT="$STATE_DIR/HALT_NEEDS_HUMAN"

POLL="${POLL:-180}"
MAX_RETRY="${MAX_RETRY:-2}"
SELF="${SLURM_JOB_ID:-none}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# tag -> "kind:sbatch_script:extra_env"
#   kind = dense | sparse   (dense is never auto-resumed on TIMEOUT)
M="/scratch/$USER/transfer_v1/oracle_masks_llama8b"
declare -A SPEC=(
  [dense_lightr1]="dense:scripts/transfer_p1_dense_dpo_lightr1_8b_deltafull_1gpu.sbatch:"
  [sparse_sp80]="sparse:scripts/transfer_p3_sparse_dpo_tulu3.sbatch:MASK_PATH=$M/oracle_dpo_tulu3_step500_sp80.pt RUN_TAG=oracle_sp80"
  [sparse_sp90]="sparse:scripts/transfer_p3_sparse_dpo_tulu3.sbatch:MASK_PATH=$M/oracle_dpo_tulu3_step500_sp90.pt RUN_TAG=oracle_sp90"
  [sparse_sp97.5]="sparse:scripts/transfer_p3_sparse_dpo_tulu3.sbatch:MASK_PATH=$M/oracle_dpo_tulu3_step500_sp97.5_src-deltalog.pt RUN_TAG=oracle_sp97.5"
  [sparse_sp99]="sparse:scripts/transfer_p3_sparse_dpo_tulu3.sbatch:MASK_PATH=$M/oracle_dpo_tulu3_step500_sp99.pt RUN_TAG=oracle_sp99"
)

# Seed the tracked job ids from the currently queued jobs.
declare -A JOB=(
  [dense_lightr1]="${DENSE_JOB:-8792090}"
  [sparse_sp80]="${SP80_JOB:-8794867}"
  [sparse_sp90]="${SP90_JOB:-8794868}"
  [sparse_sp97.5]="${SP975_JOB:-8794870}"
  [sparse_sp99]="${SP99_JOB:-8794871}"
)
declare -A RETRY=( [dense_lightr1]=0 [sparse_sp80]=0 [sparse_sp90]=0 [sparse_sp97.5]=0 [sparse_sp99]=0 )
declare -A DONE=()

state_of()   { sacct -X -n -j "$1" --format=State -P 2>/dev/null | head -1 | cut -d' ' -f1; }
elapsed_of() { sacct -X -n -j "$1" --format=ElapsedRaw -P 2>/dev/null | head -1; }
limit_of()   { sacct -X -n -j "$1" --format=TimelimitRaw -P 2>/dev/null | head -1; }

# Known-fixable environment classes only. Anything else -> halt.
classify_err() {
  local f="$1"
  [ -f "$f" ] || { echo unknown; return; }
  local t; t=$(tail -n 200 "$f" 2>/dev/null)
  case "$t" in
    *"CondaToSNonInteractiveError"*)                 echo conda_tos ;;
    *"ModuleNotFoundError"*|*"cannot import name"*)  echo env_drift ;;
    *"CUDA out of memory"*|*"OutOfMemoryError"*)     echo oom ;;
    *)                                               echo unknown ;;
  esac
}

resubmit() {
  local tag="$1" spec="${SPEC[$tag]}"
  local script="${spec#*:}"; script="${script%%:*}"
  local extra="${spec##*:}"
  # Guard: never double-submit if something with this tag is already queued or running.
  if squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep -qv "^$SELF " && \
     squeue -u "$USER" -h -t R,PD -o "%j" 2>/dev/null | grep -q "${tag##*_}"; then
    log "  guard: a job matching '$tag' is already R/PD -- not resubmitting"
    return 1
  fi
  local out
  if [ -n "$extra" ]; then
    out=$(env $extra sbatch --export=ALL,$(echo "$extra" | tr ' ' ',') "$script" 2>&1)
  else
    out=$(sbatch "$script" 2>&1)
  fi
  local new; new=$(echo "$out" | grep -oE '[0-9]+$')
  if [ -n "$new" ]; then
    JOB[$tag]="$new"; RETRY[$tag]=$(( RETRY[$tag] + 1 ))
    log "  RESUBMITTED $tag -> job $new (retry ${RETRY[$tag]}/$MAX_RETRY)"
    return 0
  fi
  log "  ERROR resubmitting $tag: $out"; return 1
}

halt() {
  local tag="$1" reason="$2"
  { echo "halted:  $(date '+%F %T')"
    echo "arm:     $tag   job ${JOB[$tag]}"
    echo "reason:  $reason"
    echo "err log: $REPO/logs/"
  } >> "$HALT"
  log "!! HALT $tag -- $reason (needs a human; wrote $HALT)"
  DONE[$tag]=halted
}

log "=== autorunner start (job $SELF) poll=${POLL}s max_retry=$MAX_RETRY ==="
log "tracking: ${JOB[*]}"

while :; do
  alldone=1
  for tag in "${!SPEC[@]}"; do
    [ -n "${DONE[$tag]:-}" ] && continue
    alldone=0
    jid="${JOB[$tag]}"; st=$(state_of "$jid")
    case "$st" in
      ""|PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED) ;;
      COMPLETED)
        log "$tag ($jid): COMPLETED"; DONE[$tag]=ok ;;
      NODE_FAIL|PREEMPTED|BOOT_FAIL)
        log "$tag ($jid): $st -- infrastructure, retrying"
        [ "${RETRY[$tag]}" -ge "$MAX_RETRY" ] && halt "$tag" "$st after $MAX_RETRY retries" || resubmit "$tag" ;;
      CANCELLED*)
        el=$(elapsed_of "$jid"); lim=$(limit_of "$jid")
        # Short of walltime and nobody asked for it -> the RC idle-GPU killer.
        if [ -n "$el" ] && [ -n "$lim" ] && [ "$el" -lt $(( lim * 60 * 80 / 100 )) ]; then
          log "$tag ($jid): CANCELLED at ${el}s of $((lim*60))s -- looks like the idle-GPU killer, retrying"
          [ "${RETRY[$tag]}" -ge "$MAX_RETRY" ] && halt "$tag" "repeated idle-GPU cancel" || resubmit "$tag"
        else
          halt "$tag" "CANCELLED near walltime -- assuming a human did it"
        fi ;;
      TIMEOUT)
        kind="${SPEC[$tag]%%:*}"
        if [ "$kind" = dense ]; then
          halt "$tag" "TIMEOUT on the dense run. Resume would skip the weight-delta callback (DPO_train.py:308) and lose the step-500 deltas this rerun exists to produce. Needs a human."
        else
          log "$tag ($jid): TIMEOUT -- sparse arm, resume is safe (auto-resume already in the script)"
          [ "${RETRY[$tag]}" -ge "$MAX_RETRY" ] && halt "$tag" "TIMEOUT after $MAX_RETRY retries" || resubmit "$tag"
        fi ;;
      OUT_OF_MEMORY)
        halt "$tag" "OUT_OF_MEMORY -- a config problem, not a transient. Retrying would burn hours." ;;
      FAILED)
        cls=$(classify_err "$(ls -t "$REPO"/logs/*"${jid}"*.err 2>/dev/null | head -1)")
        case "$cls" in
          conda_tos)
            log "$tag ($jid): conda ToS -- accepting and retrying"
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1
            [ "${RETRY[$tag]}" -ge "$MAX_RETRY" ] && halt "$tag" "conda ToS persists" || resubmit "$tag" ;;
          env_drift) halt "$tag" "import/module error -- env drift. Fix: pip install -r requirements.txt + torchao 0.17 + trl callbacks patch, then resubmit by hand." ;;
          oom)       halt "$tag" "CUDA OOM in the log -- config problem." ;;
          *)         halt "$tag" "FAILED, unrecognised error class -- read logs/*${jid}*.err" ;;
        esac ;;
      *) log "$tag ($jid): unhandled state '$st'" ;;
    esac
  done

  [ "$alldone" = 1 ] && { log "=== all arms terminal; autorunner exiting ==="; break; }

  # Self-renew before hitting our own walltime so supervision never lapses.
  if [ "$SELF" != none ]; then
    left=$(squeue -h -j "$SELF" -o "%L" 2>/dev/null)
    case "$left" in
      [0-9]:[0-9][0-9]|[0-9][0-9]:[0-9][0-9])
        log "autorunner walltime nearly up -- launching successor"
        DENSE_JOB="${JOB[dense_lightr1]}" SP80_JOB="${JOB[sparse_sp80]}" \
        SP90_JOB="${JOB[sparse_sp90]}" SP975_JOB="${JOB[sparse_sp97.5]}" \
        SP99_JOB="${JOB[sparse_sp99]}" \
        sbatch --export=ALL,DENSE_JOB,SP80_JOB,SP90_JOB,SP975_JOB,SP99_JOB \
               scripts/sweep_autorunner.sbatch >>"$LOG" 2>&1
        break ;;
    esac
  fi
  sleep "$POLL"
done

log "=== autorunner stop (job $SELF) ==="
[ -f "$HALT" ] && { echo; echo "!!! ITEMS NEEDING A HUMAN:"; cat "$HALT"; }

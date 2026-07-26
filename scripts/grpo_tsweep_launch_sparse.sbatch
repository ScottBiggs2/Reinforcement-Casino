#!/bin/bash
# Tiny launcher whose only job is to submit scripts/grpo_tsweep_sparse.sbatch once
# the multigpu QOS has room.
#
# WHY THIS EXISTS: the multigpu QOS caps a user at 8 submitted jobs, and the LoRA
# baseline array (8727274, elements 0-3, rebuttal item 2) already holds four of them.
# Submitting the 4-arm sparse array on top is rejected outright with
# QOSMaxSubmitJobPerUserLimit. Rather than cancel anything that is already queued,
# this sits on the `short` QOS — which has no submit cap — waits for the mask job,
# then retries until a slot frees.
#
# It requests one CPU and no GPU, so it costs nothing to leave parked.
#SBATCH --job-name=grpo_tsweep_launch
#SBATCH --output=logs/grpo_tsweep_launch_%j.out
#SBATCH --error=logs/grpo_tsweep_launch_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00

set -uo pipefail

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO" || exit 1
mkdir -p logs

TARGET="scripts/grpo_tsweep_sparse.sbatch"
MAX_TRIES="${MAX_TRIES:-66}"      # 66 x 10 min ~= 11 h, inside the 12 h walltime
SLEEP_S="${SLEEP_S:-600}"

echo "=========================================="
echo "Launcher for $TARGET"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "  will retry every ${SLEEP_S}s, up to $MAX_TRIES times"
echo "=========================================="

for i in $(seq 1 "$MAX_TRIES"); do
    OUT=$(sbatch --parsable "$TARGET" 2>&1)
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "[try $i] submitted sparse array: $OUT"
        echo "=========================================="
        squeue -u "$USER" -o "%.12i %.9P %.24j %.8T %R" 2>/dev/null
        exit 0
    fi
    echo "[try $i/$MAX_TRIES] $(date +%H:%M:%S) sbatch failed: $OUT"
    sleep "$SLEEP_S"
done

echo "FATAL: could not submit $TARGET after $MAX_TRIES tries"
exit 1

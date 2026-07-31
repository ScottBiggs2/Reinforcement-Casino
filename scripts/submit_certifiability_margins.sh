#!/usr/bin/env bash
# Submit the certifiability-margin DAG for one cell (model x objective x dataset), from a login node.
#
#   cache_only -> tau -> arm array (one task per arm) -> merge + figures
#
# Cells are defined below; pick one with CELL=<name>. Paths are the surviving Explorer artifacts for
# the DPO/Light-R1 runs.
#
#   cd /home/$USER/rl_casino
#   CELL=qwen3_8b_lr1  bash scripts/submit_certifiability_margins.sh
#   CELL=olmo3_7b_lr1  bash scripts/submit_certifiability_margins.sh
#
# Re-run a single stage against an existing OUT_DIR (caches are reused automatically):
#   CELL=olmo3_7b_lr1 STAGES=tau,arm,merge bash scripts/submit_certifiability_margins.sh
#
# Status:
#   CELL=olmo3_7b_lr1 bash scripts/submit_certifiability_margins.sh --status
#
# NEVER point OUT_DIR at $HOME — the magnitude caches are ~30 GB each and /home/biggs.s is at 66 GB.
# Both this script and the Python entry point refuse it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_ROOT="${TRAIN_ROOT:-${SCRATCH_USER_ROOT}/rl_casino_train}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${SCRATCH_USER_ROOT}/rl_casino_analysis/certifiability_margins}"
SBATCH_SCRIPT="${REPO_ROOT}/scripts/slurm_certifiability_margins.slurm"

CELL="${CELL:-}"
case "$CELL" in
  qwen3_8b_lr1)
    RUN_DIR="${TRAIN_ROOT}/20260601_175443_7366081"
    SUB_DIR="qwen_qwen3_8b_light_r1"
    MODEL_LABEL="Qwen3-8B"
    # Deltas exist for steps 50..250 on this run.
    MILESTONES="${MILESTONES:-50,100,150,200}"
    ;;
  olmo3_7b_lr1)
    RUN_DIR="${TRAIN_ROOT}/20260728_184922_8811836"
    SUB_DIR="allenai_olmo_3_7b_instruct_light_r1"
    MODEL_LABEL="Olmo-3-7B-Instruct"
    # Deltas exist for steps 50..500; the paper's grid is 50..200.
    MILESTONES="${MILESTONES:-50,100,150,200}"
    ;;
  *)
    echo "ERROR: set CELL to one of: qwen3_8b_lr1 olmo3_7b_lr1" >&2
    exit 1
    ;;
esac

export INITIAL_MODEL="${RUN_DIR}/deltas/${SUB_DIR}/base_state.pt"
export FINAL_MODEL="${RUN_DIR}/checkpoints/${SUB_DIR}/checkpoint-500"
export DELTA_LOG_DIR="${RUN_DIR}/deltas/${SUB_DIR}"
export OUT_DIR="${OUT_DIR:-${ANALYSIS_ROOT}/${CELL}}"
export MILESTONES
export MODEL_LABEL
export DATASET_LABEL="${DATASET_LABEL:-Light-R1}"
export OBJECTIVE_LABEL="${OBJECTIVE_LABEL:-DPO}"
export SPARSITIES="${SPARSITIES:-97.5}"
export RANDOM_SEED="${RANDOM_SEED:-42}"

# Arms: oracle, one per milestone, and the random baseline. Order fixes the array indices.
ARMS="oracle"
IFS=',' read -r -a _MS <<< "$MILESTONES"
for m in "${_MS[@]}"; do ARMS="${ARMS},warm_k${m// /}"; done
ARMS="${ARMS},random_seed${RANDOM_SEED}"
export CERT_ARMS="$ARMS"
IFS=',' read -r -a _ARMS <<< "$ARMS"
N_ARMS="${#_ARMS[@]}"

if [ "${1:-}" = "--status" ]; then
  echo "cell=${CELL}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "--- magnitude caches"
  ls -la "${OUT_DIR}/magnitude_caches" 2>/dev/null || echo "  (none)"
  echo "--- tau"
  ls -la "${OUT_DIR}/certifiability_tau.json" 2>/dev/null || echo "  (none)"
  echo "--- arm shards (${N_ARMS} expected)"
  ls -la "${OUT_DIR}/arm_shards" 2>/dev/null || echo "  (none)"
  echo "--- final artifacts"
  ls -la "${OUT_DIR}"/certifiability_{summary.csv,margins.npz,diagnostics.json} 2>/dev/null || echo "  (none)"
  echo "--- queue"
  squeue -u "${USER}" -n certmargin -o "%.12i %.10P %.12j %.8T %.10M %R" 2>/dev/null || true
  exit 0
fi

for p in "$INITIAL_MODEL" "$DELTA_LOG_DIR" "$FINAL_MODEL"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing artifact: $p" >&2
    exit 1
  fi
done

case "$(cd "$(dirname "$OUT_DIR")" 2>/dev/null && pwd 2>/dev/null || echo /nonexistent)/$(basename "$OUT_DIR")" in
  "${HOME}"|"${HOME}"/*)
    echo "REFUSING: OUT_DIR=${OUT_DIR} is under \$HOME. Use ${ANALYSIS_ROOT}/..." >&2
    exit 1
    ;;
esac
mkdir -p "$OUT_DIR" logs

STAGES="${STAGES:-cache_only,tau,arm,merge}"
want() { case ",${STAGES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# cache_only streams every delta file once (5 x ~15-33 GB); tau and arm hold two state dicts plus one
# magnitude cache in RAM. merge touches no checkpoint.
SB_CACHE="${SB_CACHE:---mem=192G --time=08:00:00 --cpus-per-task=8}"
SB_TAU="${SB_TAU:---mem=192G --time=12:00:00 --cpus-per-task=16}"
SB_ARM="${SB_ARM:---mem=192G --time=08:00:00 --cpus-per-task=16}"
SB_MERGE="${SB_MERGE:---mem=32G --time=01:00:00 --cpus-per-task=2}"
SB_EXTRA="${SB_EXTRA:-}"

echo "cell=${CELL}"
echo "  theta^0 : ${INITIAL_MODEL}"
echo "  theta^T : ${FINAL_MODEL}"
echo "  deltas  : ${DELTA_LOG_DIR}"
echo "  OUT_DIR : ${OUT_DIR}"
echo "  arms    : ${ARMS}  (array 0-$((N_ARMS - 1)))"
echo "  rho     : ${SPARSITIES}"
echo "  stages  : ${STAGES}"

DEP=""
maybe_dep() { [ -n "$DEP" ] && echo "--dependency=afterok:${DEP}" || echo ""; }

if want cache_only; then
  # shellcheck disable=SC2046,SC2086
  J=$(sbatch --parsable $SB_EXTRA $SB_CACHE $(maybe_dep) \
      --job-name=certcache --export=ALL,CERT_STAGE=cache_only "$SBATCH_SCRIPT")
  echo "cache_only  job=${J}"
  DEP="$J"
fi

if want tau; then
  # shellcheck disable=SC2046,SC2086
  J=$(sbatch --parsable $SB_EXTRA $SB_TAU $(maybe_dep) \
      --job-name=certtau --export=ALL,CERT_STAGE=tau "$SBATCH_SCRIPT")
  echo "tau         job=${J}"
  DEP="$J"
fi

if want arm; then
  # shellcheck disable=SC2046,SC2086
  J=$(sbatch --parsable $SB_EXTRA $SB_ARM $(maybe_dep) --array="0-$((N_ARMS - 1))" \
      --job-name=certarm --export=ALL,CERT_STAGE=arm "$SBATCH_SCRIPT")
  echo "arm array   job=${J}"
  DEP="$J"
fi

if want merge; then
  # shellcheck disable=SC2046,SC2086
  J=$(sbatch --parsable $SB_EXTRA $SB_MERGE $(maybe_dep) \
      --job-name=certmerge --export=ALL,CERT_STAGE=merge "$SBATCH_SCRIPT")
  echo "merge       job=${J}"
fi

echo
echo "Artifacts land in ${OUT_DIR} (CSV/NPZ/JSON + figures/)."
echo "Copy ONLY those off scratch — never magnitude_caches/ or arm_shards/."

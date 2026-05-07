#!/usr/bin/env bash
# Login-node orchestrator: submit (1) CPU mask prefetch — optional but recommended —
# and (2) independent GPU benchmark shards per nonzero sparsity, plus a dense-only shard for "0".
#
# Layout:
#   RUN_ROOT/masks/*.pt          — shared cache (H200_BSR_MASK_CACHE=RUN_ROOT)
#   RUN_ROOT/bench_spXXpYY/      — one Slurm job per sparse level (dense baseline omitted)
#   RUN_ROOT/bench_dense/       — dense baseline only (empty --benchmark_sparsities)
#
# After all jobs finish, merge artifacts (see bottom) or run:
#   python scripts/merge_bsr_shard_csvs.py --run-root "$RUN_ROOT"
#
# --- Explorer (Northeastern RC): sync branch + launch (paste on login node) ---
#
#   # One block: checkout cav_fixes, env, HF token if needed, run orchestrator (GPU benches use multigpu queue).
#   cd /scratch/${USER}/rl_casino/RL_Casino_Working_Branch   # or your synced repo path containing scripts/
#   git fetch origin && git checkout cav_fixes && git pull --ff-only origin cav_fixes
#   mkdir -p logs
#   [ -f ~/.cache/huggingface/token ] && export HF_TOKEN="$(tr -d '\r\n' < ~/.cache/huggingface/token)"
#   export SCRATCH_USER_ROOT="/scratch/${USER}"
#   unset TRAIN_ENV TRAIN_PY
#   chmod +x scripts/h200_bsr_orchestrate.sh
#   ./scripts/h200_bsr_orchestrate.sh
#
# Optional overrides before ./scripts/h200_bsr_orchestrate.sh:
#
#   export RUN_ID="bsr_orch_${USER}_$(date +%Y%m%d_%H%M%S)"
#   export BENCHMARK_SPARSITIES=0,50,75,90,99.75
#   export H200_BSR_SLURM_PARTITION=multigpu   # default (shorter queue); use gpu to force gpu partition instead
#   export H200_BSR_BENCH_AFTER_MASK=0         # GPU jobs do not wait on CPU prefetch (see troubleshooting below)
#   export H200_BSR_SKIP_PREFETCH=1            # do not submit the short-partition prefetch job at all
#
# Single manual GPU bench shard (same multigpu default as embedded in scripts/h200_sparse_dpo_bsr_benchmark.sh):
#
#   cd /scratch/${USER}/rl_casino/RL_Casino_Working_Branch && mkdir -p logs
#   export SCRATCH_USER_ROOT="/scratch/${USER}" H200_BSR_OUT="/scratch/${USER}/rl_casino_h200_bsr/my_shard" RUN_ID=test1
#   sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
#
# After shards finish — merge CSV + theory then report (merged dir matches report_h200_speed_ablation.py --run-dir):
#
#   RUN_ROOT=/scratch/${USER}/rl_casino_h200_bsr/<YOUR_RUN_ID_FROM_ORCH_STDOUT>
#   python scripts/merge_bsr_shard_csvs.py --run-root "$RUN_ROOT"
#   python scripts/report_h200_speed_ablation.py --run-dir "${RUN_ROOT}/merged"
#
# Usage (from repo root):
#   export SCRATCH_USER_ROOT=/scratch/$USER
#   ./scripts/h200_bsr_orchestrate.sh
#
# Optional:
#   export RUN_ID=my_experiment
#   export BENCHMARK_SPARSITIES=0,50,75,90,99.75   # default includes 0 → dense shard
#   export H200_BSR_BENCH_AFTER_MASK=0             # do not wait for mask job (default 1)
#
# --- If ``short`` prefetch blocks everything (GPU jobs stuck PD / Dependency) ---
#
# 1) Start GPU shards immediately (recommended): masks are still written under RUN_ROOT/masks/ **inside**
#    each bench job on cache miss — duplicate CPU mask generation per shard, but no idle GPU: training
#    uses the H200 for hours.
#       export H200_BSR_BENCH_AFTER_MASK=0
#       ./scripts/h200_bsr_orchestrate.sh
#
# 2) Do **not** submit prefetch on ``multigpu`` without training: mask generation is CPU/meta-model
#    work; holding an allocated GPU idle for tens of minutes violates typical Explorer GPU-use policies.
#
# 3) Precompute masks once when ``short`` is quiet (standalone prefetch), then reuse RUN_ROOT for later
#    bench submits; or generate masks from another machine and rsync RUN_ROOT/masks/ onto scratch.
#
# 4) Cancel stuck orchestrator + relaunch ASAP (reuse same RUN_ID so RUN_ROOT matches):
#       scancel <prefetch_jid> <bench_jid...>
#       export RUN_ID=<same_as_before>  H200_BSR_BENCH_AFTER_MASK=0  H200_BSR_SKIP_PREFETCH=1
#       ./scripts/h200_bsr_orchestrate.sh
#
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
case "${H200_BSR_OUT:-}" in
  /rl_casino_h200_bsr/*)
    echo "WARNING: Ignoring invalid H200_BSR_OUT=${H200_BSR_OUT}" >&2
    unset H200_BSR_OUT
    ;;
esac

ORCH_BASE="${H200_BSR_ORCH_BASE:-${SCRATCH_USER_ROOT}/rl_casino_h200_bsr}"
export RUN_ID="${RUN_ID:-orch_${USER}_${HOSTNAME:-local}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${ORCH_BASE}/${RUN_ID}"
mkdir -p "${RUN_ROOT}/masks" logs

export BENCHMARK_SPARSITIES="${BENCHMARK_SPARSITIES:-0,50,75,90,99.75}"
# Bench jobs: multigpu + single H200 (shorter queue on Explorer vs partition=gpu).
H200_BSR_SLURM_PARTITION="${H200_BSR_SLURM_PARTITION:-multigpu}"

SPARSE_MASK_LIST=()
HAS_ZERO=0
IFS=',' read -ra _RAW_LEVELS <<< "${BENCHMARK_SPARSITIES}"
for raw in "${_RAW_LEVELS[@]}"; do
  s="${raw//[[:space:]]/}"
  [[ -z "${s}" ]] && continue
  if [[ "${s}" == "0" || "${s}" == "0.0" ]]; then
    HAS_ZERO=1
  else
    SPARSE_MASK_LIST+=("${s}")
  fi
done

JOIN_SPARSE_MASK=$(
  IFS=,
  echo "${SPARSE_MASK_LIST[*]}"
)

echo "RUN_ROOT=${RUN_ROOT}"
echo "H200_BSR_SLURM_PARTITION=${H200_BSR_SLURM_PARTITION}  (passed to sbatch for GPU benchmark shards)"
echo "Configured sparsities: ${BENCHMARK_SPARSITIES}"
echo "Dense-only shard (0%%): $([[ ${HAS_ZERO} -eq 1 ]] && echo yes || echo no)"
echo "Sparse prefetch / bench levels: ${JOIN_SPARSE_MASK:-<none>}"

MASK_JID=""
if ((${#SPARSE_MASK_LIST[@]} > 0)); then
  if [[ "${H200_BSR_SKIP_PREFETCH:-0}" != "0" ]]; then
    echo "Skipping mask prefetch Slurm job (H200_BSR_SKIP_PREFETCH=${H200_BSR_SKIP_PREFETCH}); benches build masks on demand under RUN_ROOT/masks/."
  else
    MASK_JID=$(sbatch --parsable \
      --export=ALL,H200_BSR_MASK_CACHE="${RUN_ROOT}",BENCHMARK_SPARSITIES="${JOIN_SPARSE_MASK}",H200_BSR_PREFETCH_OUT="${RUN_ROOT}/prefetch_log" \
      "${REPO_ROOT}/scripts/h200_bsr_prefetch_masks.slurm")
    echo "Submitted mask prefetch job: ${MASK_JID}"
  fi
else
  echo "No nonzero sparsities; skipping mask prefetch job."
fi

DEP_PREFIX=()
if [[ "${H200_BSR_BENCH_AFTER_MASK:-1}" != "0" ]] && [[ -n "${MASK_JID}" ]]; then
  DEP_PREFIX=(--dependency=afterok:"${MASK_JID}")
fi

BENCH_JIDS=()

if ((${#SPARSE_MASK_LIST[@]} > 0)); then
  for s in "${SPARSE_MASK_LIST[@]}"; do
    tag="${s//./p}"
    out="${RUN_ROOT}/bench_sp${tag}"
    jid=$(sbatch --parsable "${DEP_PREFIX[@]}" \
      --partition="${H200_BSR_SLURM_PARTITION}" \
      --export=ALL,H200_BSR_OUT="${out}",BENCHMARK_SPARSITIES="${s}",H200_BSR_SKIP_DENSE=1,H200_BSR_MASK_CACHE="${RUN_ROOT}" \
      "${REPO_ROOT}/scripts/h200_sparse_dpo_bsr_benchmark.sh")
    echo "Submitted bench sparse s=${s} → ${out}  job=${jid}"
    BENCH_JIDS+=("${jid}")
  done
fi

if [[ "${HAS_ZERO}" -eq 1 ]]; then
  out="${RUN_ROOT}/bench_dense"
  jid=$(sbatch --parsable "${DEP_PREFIX[@]}" \
    --partition="${H200_BSR_SLURM_PARTITION}" \
    --export=ALL,H200_BSR_OUT="${out}",BENCHMARK_SPARSITIES="",H200_BSR_SKIP_DENSE=0,H200_BSR_MASK_CACHE="${RUN_ROOT}" \
    "${REPO_ROOT}/scripts/h200_sparse_dpo_bsr_benchmark.sh")
  echo "Submitted bench dense-only → ${out}  job=${jid}"
  BENCH_JIDS+=("${jid}")
fi

echo ""
echo "RUN_ROOT=${RUN_ROOT}"
echo "Shared masks: ${RUN_ROOT}/masks/"
echo "To merge CSV + theory after jobs complete:"
echo "  python ${REPO_ROOT}/scripts/merge_bsr_shard_csvs.py --run-root ${RUN_ROOT}"
echo ""
echo "Manual CSV merge (first row is header; pick one canonical header from a shard):"
echo "  mkdir -p ${RUN_ROOT}/merged && f=(${RUN_ROOT}/bench_*/benchmark_training_log.csv)"
echo "  head -1 \"\${f[0]}\" > ${RUN_ROOT}/merged/benchmark_training_log.csv"
echo "  for c in \"\${f[@]}\"; do tail -n +2 \"\$c\"; done >> ${RUN_ROOT}/merged/benchmark_training_log.csv"
echo ""
echo "Submitted job IDs: mask=${MASK_JID:-n/a}  benches=${BENCH_JIDS[*]}"

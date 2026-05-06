#!/usr/bin/env bash
# Mask score-gap analysis: oracle vs warm magnitude at deltas 50/100/150/200 (+ random baseline).
# Writes CSV/NPZ/gap diagnostics + matplotlib overlays under OUT_DIR/figures.
#
# Drop-in (Explorer defaults; HF + repo path unchanged from last run):
#
#   cd /home/$USER/rl_casino
#   export HF_TOKEN="$(tr -d '\n\r' < ~/.cache/huggingface/token)"   # optional if --export=ALL
#   export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"               # cert tau / margins (alias: CERT_SPARSITY_PERCENT)
#   export CERT_MIN_LAYER_KEEP_RATIO="${CERT_MIN_LAYER_KEEP_RATIO:-0}"
#   export CERT_MATCH_TIE_BREAK="${CERT_MATCH_TIE_BREAK:-1}"         # set 0 to pass --cert_no_match_tie_break
#   export CHECKPOINT_DTYPE=bfloat16   # bf16 halves RAM vs fp32; use float32 if you need bit-identical baselines
#   export CERT_GLOBAL_MODE=stream     # stream tau/margins without full-model flat buffers (recommended)
#   sbatch scripts/slurm_mask_score_gap_light_r1.slurm
#
# Compare Theorem-3 global τ vs hybrid (per-layer floors + global top-R), same OUT_DIR parent, different leaf dirs:
#
#   export CERT_GLOBAL_MODE=stream CERT_TAU_RULE=global CERT_HYBRID_MIN_LAYER_KEEP_RATIO=0
#   export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_light_r1/${SLURM_JOB_ID:-global}
#   sbatch scripts/slurm_mask_score_gap_light_r1.slurm
#
#   export CERT_GLOBAL_MODE=stream CERT_TAU_RULE=hybrid_global_phase CERT_HYBRID_MIN_LAYER_KEEP_RATIO=0.02
#   export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_light_r1/${SLURM_JOB_ID:-hybrid}
#   sbatch scripts/slurm_mask_score_gap_light_r1.slurm
#
# (Use two real job IDs or set OUT_DIR explicitly before each sbatch so runs do not overwrite.)
#
# If the job dies with Bash "Killed" and no Python traceback, it is usually cgroup/OOM (not TRAIN_PY).
# Confirm vs your Slurm memory request:
#   sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,ReqMem,MaxRSS,Elapsed -P
# (MaxRSS is batch-step RSS; compare to #SBATCH --mem.) Raise --mem or use a highmem partition if needed.
#
# Copy out (after job):
#   mkdir -p ~/rl_casino/results/mask_score_gap_${JOBID} && \
#   cp -a /scratch/$USER/rl_casino_analysis/mask_score_gap_light_r1/${JOBID}/. ~/rl_casino/results/mask_score_gap_${JOBID}/
# (Or rely on job-end mirror to ~/rl_casino/results/mask_score_gap_<jobid>/ — see bottom of script.)
#
# Submit ONLY with sbatch (this file is not chmod +x; do not run it as ./scripts/... or you get
# "Permission denied"). Always: sbatch path/to/this/file.slurm
#
# cd to the GIT REPO ROOT (the directory that contains scripts/ and src/), then:
#   sbatch scripts/slurm_mask_score_gap_light_r1.slurm
#
# After the job, a copy is written to ${HOME}/rl_casino/results/mask_score_gap_<jobid>/ (override root
# with RESULTS_MIRROR_ROOT, or set RESULTS_MIRROR_DISABLE=1 to skip).
#
# Gated Llama: pass HF token on ONE line (multi-line backslash pastes often break and the next
# line runs alone as a command). Example — replace YOUR_TOKEN, use real paths if not defaults:
#
#   cd /scratch/$USER/path/to/RL_Casino_Working_Branch && sbatch --export=ALL,HF_TOKEN=YOUR_TOKEN,TRAIN_ENV=/scratch/$USER/conda_envs/rl_casino scripts/slurm_mask_score_gap_light_r1.slurm
#
# Optional overrides (still one line, comma-separated after ALL):
#   ...,CKPT500_DIR=/scratch/$USER/.../checkpoint-500,DELTA_LOG_DIR=/scratch/$USER/.../deltas/...,OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_gap1,VERIFY_DELTAS_DIR=/path/with/deltas_step_500.pt
#   ...,SPARSITY_PERCENT=97.5,CERT_MIN_LAYER_KEEP_RATIO=0,CERT_MATCH_TIE_BREAK=1,CERT_TAU_RULE=global,CERT_HYBRID_MIN_LAYER_KEEP_RATIO=0.02
#
# If TRAIN_PY errors with /conda_envs/... (no /scratch/$USER): your shell exported TRAIN_ENV wrong.
#   unset TRAIN_ENV
#   sbatch scripts/slurm_mask_score_gap_light_r1.slurm
#
# If the shell "hangs" before any Slurm message: check for an unclosed quote in --export=, or run
#   sbatch scripts/slurm_mask_score_gap_light_r1.slurm
# alone first (uses HF_TOKEN already exported in your shell: export HF_TOKEN=...).
#
# Local smoke (tiny subset of layers; still loads full checkpoints — use HPC for real runs):
#   mkdir -p /tmp/mask_gap_smoke && \\
#   python src/analysis/mask_score_gap_analysis.py \\
#     --initial_model meta-llama/Llama-3.1-8B-Instruct \\
#     --final_model "${CKPT500_DIR}" \\
#     --delta_log_dir "${DELTA_LOG_DIR}" \\
#     --magnitude_milestones 50,100,150,200 \\
#     --out_dir /tmp/mask_gap_smoke \\
#     --max_keys 3 --histogram_bins 256 \\
#     --verify_deltas_dir "${DELTA_LOG_DIR}"
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Full 8B×2 + fp32 milestone caches + histograms can exceed 128G; 512G is a safer default on Explorer.
# If the scheduler rejects this request, switch to your site's highmem partition or lower --mem and rely on CERT_GLOBAL_MODE=stream + bf16 weights.
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --job-name=mask_score_gap
#SBATCH --output=logs/mask_score_gap_%j.out
#SBATCH --error=logs/mask_score_gap_%j.err
# Propagate your login-shell env (incl. HF_TOKEN) into the job when you export before sbatch:
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
_DEFAULT_TRAIN_ENV="${SCRATCH_USER_ROOT}/conda_envs/rl_casino"
# --export=ALL can inherit a broken TRAIN_ENV from the shell (e.g. /conda_envs/...); prefer default if invalid.
TRAIN_ENV="${TRAIN_ENV:-${_DEFAULT_TRAIN_ENV}}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "WARN: TRAIN_PY missing at ${TRAIN_PY}; overriding TRAIN_ENV → ${_DEFAULT_TRAIN_ENV}" >&2
  TRAIN_ENV="${_DEFAULT_TRAIN_ENV}"
  TRAIN_PY="${TRAIN_ENV}/bin/python"
fi
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  echo "  Fix: export TRAIN_ENV=/scratch/\${USER}/conda_envs/rl_casino or unset TRAIN_ENV before sbatch." >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-${SCRATCH_USER_ROOT}/hf_cache/hub}"

# Slurm clears env unless --export=ALL; hf auth login stores token on shared $HOME:
_HF_DEFAULT_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  if [ -n "${HF_TOKEN_FILE:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
    export HF_TOKEN="$(tr -d '\n\r' < "${HF_TOKEN_FILE}")"
  elif [ -f "${_HF_DEFAULT_TOKEN_FILE}" ]; then
    export HF_TOKEN="$(tr -d '\n\r' < "${_HF_DEFAULT_TOKEN_FILE}")"
  fi
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set (length ${#HF_TOKEN})"
else
  echo "WARN: HF_TOKEN unset and no readable token at HF_TOKEN_FILE or ${_HF_DEFAULT_TOKEN_FILE}" >&2
fi

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CKPT500_DIR="${CKPT500_DIR:-${SCRATCH_USER_ROOT}/rl_casino_train/dpo5k_dense_light-r1/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500}"
DELTA_LOG_DIR="${DELTA_LOG_DIR:-${SCRATCH_USER_ROOT}/rl_casino_train/dpo5k_dense_light-r1/deltas/meta_llama_llama_3_1_8b_instruct_light_r1}"
OUT_DIR="${OUT_DIR:-${SCRATCH_USER_ROOT}/rl_casino_analysis/mask_score_gap_light_r1/${SLURM_JOB_ID:-local}}"
MAGNITUDE_MILESTONES="${MAGNITUDE_MILESTONES:-50,100,150,200}"
FORCE_MAG_CACHE_REBUILD="${FORCE_MAG_CACHE_REBUILD:-0}"
RANDOM_SEEDS="${RANDOM_SEEDS:-}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-2048}"
MAX_KEYS="${MAX_KEYS:-}"
# Optional: directory containing deltas_step_500.pt (omit if that snapshot does not exist yet).
VERIFY_DELTAS_DIR="${VERIFY_DELTAS_DIR:-}"
# Certifiability margins (Theorem 3): global tau at sparsity rho; see mask_score_gap_gap_diagnostics.json
SPARSITY_PERCENT="${CERT_SPARSITY_PERCENT:-${SPARSITY_PERCENT:-97.5}}"
CERT_MIN_LAYER_KEEP_RATIO="${CERT_MIN_LAYER_KEEP_RATIO:-0}"
CERT_MATCH_TIE_BREAK="${CERT_MATCH_TIE_BREAK:-1}"
CHECKPOINT_DTYPE="${MASK_GAP_CHECKPOINT_DTYPE:-${CHECKPOINT_DTYPE:-bfloat16}}"
CERT_GLOBAL_MODE="${CERT_GLOBAL_MODE:-stream}"
CERT_TAU_RULE="${CERT_TAU_RULE:-global}"
CERT_HYBRID_MIN_LAYER_KEEP_RATIO="${CERT_HYBRID_MIN_LAYER_KEEP_RATIO:-0}"

mkdir -p "${OUT_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "SCRATCH_USER_ROOT=${SCRATCH_USER_ROOT}"
echo "TRAIN_ENV=${TRAIN_ENV}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "CKPT500_DIR=${CKPT500_DIR}"
echo "DELTA_LOG_DIR=${DELTA_LOG_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAGNITUDE_MILESTONES=${MAGNITUDE_MILESTONES}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT}"
echo "CERT_MIN_LAYER_KEEP_RATIO=${CERT_MIN_LAYER_KEEP_RATIO}"
echo "CERT_MATCH_TIE_BREAK=${CERT_MATCH_TIE_BREAK}"
echo "CHECKPOINT_DTYPE=${CHECKPOINT_DTYPE}"
echo "CERT_GLOBAL_MODE=${CERT_GLOBAL_MODE}"
echo "CERT_TAU_RULE=${CERT_TAU_RULE}"
echo "CERT_HYBRID_MIN_LAYER_KEEP_RATIO=${CERT_HYBRID_MIN_LAYER_KEEP_RATIO}"

ARGS=(
  src/analysis/mask_score_gap_analysis.py
  --initial_model "${BASE_MODEL}"
  --final_model "${CKPT500_DIR}"
  --delta_log_dir "${DELTA_LOG_DIR}"
  --magnitude_milestones "${MAGNITUDE_MILESTONES}"
  --out_dir "${OUT_DIR}"
  --histogram_bins "${HISTOGRAM_BINS}"
  --sparsity_percent "${SPARSITY_PERCENT}"
  --cert_min_layer_keep_ratio "${CERT_MIN_LAYER_KEEP_RATIO}"
  --checkpoint_dtype "${CHECKPOINT_DTYPE}"
  --cert_global_mode "${CERT_GLOBAL_MODE}"
  --cert_tau_rule "${CERT_TAU_RULE}"
  --cert_hybrid_min_layer_keep_ratio "${CERT_HYBRID_MIN_LAYER_KEEP_RATIO}"
)
if [ "${CERT_MATCH_TIE_BREAK}" = "0" ] || [ "${CERT_MATCH_TIE_BREAK}" = "false" ]; then
  ARGS+=( --cert_no_match_tie_break )
fi
if [ "${FORCE_MAG_CACHE_REBUILD}" = "1" ]; then
  ARGS+=( --force_magnitude_cache_rebuild )
fi
if [ -n "${VERIFY_DELTAS_DIR}" ]; then
  ARGS+=( --verify_deltas_dir "${VERIFY_DELTAS_DIR}" )
fi
if [ -n "${RANDOM_SEEDS}" ]; then
  ARGS+=( --random_seeds "${RANDOM_SEEDS}" )
fi
if [ -n "${MAX_KEYS}" ]; then
  ARGS+=( --max_keys "${MAX_KEYS}" )
fi
"${TRAIN_PY}" "${ARGS[@]}"

"${TRAIN_PY}" scripts/report_mask_score_gap_plots.py --analysis-dir "${OUT_DIR}"

echo "Done. Artifacts: ${OUT_DIR} (figures in ${OUT_DIR}/figures)"

# Optional copy to $HOME for quick browsing (disable with RESULTS_MIRROR_DISABLE=1)
if [ "${RESULTS_MIRROR_DISABLE:-0}" != "1" ]; then
  _MIRROR_ROOT="${RESULTS_MIRROR_ROOT:-${HOME}/rl_casino/results}"
  _MIRROR="${_MIRROR_ROOT}/mask_score_gap_${SLURM_JOB_ID}"
  mkdir -p "${_MIRROR}"
  cp -a "${OUT_DIR}/." "${_MIRROR}/"
  echo "Mirrored to ${_MIRROR}"
fi

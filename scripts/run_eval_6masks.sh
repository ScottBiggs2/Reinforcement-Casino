#!/bin/bash
# Orchestrator: save 6 masked checkpoints + submit 7 eval jobs (dense + 6 masks).
# Uses EVAL_LIMIT=300 by default for a fast "trend" run (~1.5h per config).
#
# Run on the login node (not via sbatch — this script submits sbatch jobs).
#
#   bash scripts/run_eval_6masks.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/xie.yiyi/Reinforcement-Casino}"
cd "$REPO_ROOT"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EVAL_LIMIT="${EVAL_LIMIT:-300}"
CKPT_ROOT="${CKPT_ROOT:-/scratch/xie.yiyi/masked_ckpts}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "FATAL: HF token not found at $HF_TOKEN_FILE" >&2
    exit 1
fi
HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export HF_TOKEN

mkdir -p logs

# (label, mask_path) — same 6 you just probed / erank+CKA'd.
MASKS=(
  "Random-DPO    /scratch/xie.yiyi/rl_casino_masks/llama8b/random_baseline_dpo_sp97.5_seed42.pt"
  "Cold-CAV-DPO  /scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt"
  "Oracle-DPO    /scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt"
  "Random-GRPO   /scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/random_baseline_grpo_sp97.5_seed42.pt"
  "Cold-CAV-GRPO /scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_grpo.pt"
  "Oracle-GRPO   /scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt"
)

# ---------------------------------------------------------------------------
# Stage 1: save_pretrained for each mask (sequential; needs one H200)
# ---------------------------------------------------------------------------
SAVE_LOG="logs/save_masked_ckpts_$(date +%Y%m%d_%H%M%S).out"
cat > /tmp/save_masked_ckpts.sh <<EOF
#!/bin/bash
#SBATCH --job-name=save_masked
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=${SAVE_LOG/.out/_%j.out}
#SBATCH --error=${SAVE_LOG/.out/_%j.err}

set -euo pipefail

CUDA_LIB=\$(dirname "\$(find /usr /opt /lib64 /lib -name 'libcuda.so.1' 2>/dev/null | head -n1)")
if [ -n "\$CUDA_LIB" ] && [ "\$CUDA_LIB" != "." ]; then
    export LD_LIBRARY_PATH="\${CUDA_LIB}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
fi

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
if [ -f "\$CONDA_SH" ]; then
    source "\$CONDA_SH"
    if [ -d "/scratch/xie.yiyi/conda_envs/rl_casino" ]; then
        conda activate "/scratch/xie.yiyi/conda_envs/rl_casino"
    elif [ -d "/home/xie.yiyi/.conda/envs/rl_casino" ]; then
        conda activate "/home/xie.yiyi/.conda/envs/rl_casino"
    fi
fi

cd "$REPO_ROOT"
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export HF_TOKEN="$HF_TOKEN"

mkdir -p "$CKPT_ROOT"

EOF

for entry in "${MASKS[@]}"; do
    label="$(awk '{print $1}' <<< "$entry")"
    mpath="$(awk '{print $2}' <<< "$entry")"
    cat >> /tmp/save_masked_ckpts.sh <<EOF
echo "=== saving $label ==="
python src/evaluation/apply_mask_and_save.py \\
    --base_model "$BASE_MODEL" \\
    --mask "$mpath" \\
    --output_dir "$CKPT_ROOT/$label"
EOF
done

SAVE_JOB=$(sbatch --parsable /tmp/save_masked_ckpts.sh)
echo "[orchestrator] Stage-1 save_pretrained job: $SAVE_JOB"

# ---------------------------------------------------------------------------
# Stage 2: eval (dense + each saved ckpt), each job depends on save_masked.
# All results land in ${RESULTS_ROOT}/<Label>/ — one dir per config.
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_ROOT="results/eval_6masks_${TS}"
MANIFEST="${RESULTS_ROOT}/manifest.tsv"
mkdir -p "$RESULTS_ROOT"
printf 'label\tjob_id\toutput_dir\n' > "$MANIFEST"

EVAL_JOBS=()

# Dense uses HF model ID directly, no dependency on the save job.
DENSE_OUT="${RESULTS_ROOT}/Dense"
mkdir -p "$DENSE_OUT"
DENSE_JOB=$(sbatch --parsable \
    --export=ALL,HF_TOKEN="$HF_TOKEN",EVAL_LIMIT="$EVAL_LIMIT" \
    scripts/run_evals_slurm.sh \
        --model_path "$BASE_MODEL" --trust_remote_code \
        --output_dir "$DENSE_OUT")
echo "[orchestrator] Dense eval job: $DENSE_JOB  → $DENSE_OUT"
EVAL_JOBS+=("Dense:$DENSE_JOB")
printf 'Dense\t%s\t%s\n' "$DENSE_JOB" "$DENSE_OUT" >> "$MANIFEST"

for entry in "${MASKS[@]}"; do
    label="$(awk '{print $1}' <<< "$entry")"
    ckpt="$CKPT_ROOT/$label"
    out="${RESULTS_ROOT}/${label}"
    mkdir -p "$out"
    jid=$(sbatch --parsable \
        --dependency=afterok:$SAVE_JOB \
        --export=ALL,HF_TOKEN="$HF_TOKEN",EVAL_LIMIT="$EVAL_LIMIT" \
        scripts/run_evals_slurm.sh \
            --model_path "$ckpt" --trust_remote_code \
            --output_dir "$out")
    echo "[orchestrator] $label eval job: $jid  → $out"
    EVAL_JOBS+=("$label:$jid")
    printf '%s\t%s\t%s\n' "$label" "$jid" "$out" >> "$MANIFEST"
done

echo ""
echo "[orchestrator] All jobs submitted; manifest: $MANIFEST"
printf '  %s\n' "${EVAL_JOBS[@]}"
echo ""
echo "Stage-1 save_pretrained is gating the 6 masked evals; Dense runs independently."
echo "Monitor:  squeue -u \$USER"
echo "Aggregate: python src/analysis/plot_eval_comparison.py --manifest $MANIFEST"

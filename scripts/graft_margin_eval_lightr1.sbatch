#!/bin/bash
#SBATCH --job-name=graft_margin
#SBATCH --output=logs/graft_margin_%j.out
#SBATCH --error=logs/graft_margin_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=d1025

# B: held-out DPO implicit-reward margin under grafting (Panigrahi ICML 2023).
#
# Conditions (policy vs ref=base, beta 0.1, protocol frozen at the arm0
# settings: holdout_tail 2000, n_eval 500, max_length 1024):
#   light-r1 (in-task):
#     dense           theta_ft                      (ceiling)
#     graft_oracle    theta_base + M.Delta          (sufficiency)
#     graft_anti      theta_base + (1-M).Delta      (necessity)
#     graft_random42  theta_base + M_rand.Delta     (specificity; matched 291-tensor random)
#     graft_a05       theta_base + 0.5*M.Delta      (dose-response point)
#   tulu3 (cross-dataset rider):
#     dense, graft_oracle                            (transfer of the grafted update)
#
# base_llama margin 0.0000 on light-r1 is already on record (job 8727254_0);
# not re-run. HARDWARE PINNED to a100: v100-pcie has no bf16 (see the 8786294
# post-mortem in preference_eval_lora_arm2.sbatch).

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi
REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO" || exit 1
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_USER_ROOT:-/scratch/$USER}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"

BASE=meta-llama/Llama-3.1-8B-Instruct
DENSE=$SCRATCH_ROOT/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500
GRAFTS=$SCRATCH_ROOT/graft_ckpts_lightr1
OUT=$SCRATCH_ROOT/rebuttal_analysis/graft_margin
mkdir -p "$OUT"

for d in graft_oracle_a1 graft_anti_a1 graft_random42_a1 graft_oracle_a05; do
    test -s "$GRAFTS/$d/graft_meta.json" || { echo "FATAL: missing graft $d"; exit 1; }
done

FAILED=0
run_eval () { # tag model_path dataset
    local tag="$1" model="$2" ds="$3"
    local out_json="$OUT/${tag}.json"
    if [ -s "$out_json" ]; then echo "[skip] $tag"; return; fi
    echo "======== $tag  ($ds)  $(date) ========"
    python src/evaluation/preference_eval.py \
        --model "$model" \
        --ref_model "$BASE" \
        --eval_dataset "$ds" \
        --holdout_tail 2000 \
        --n_eval 500 \
        --trained_on light-r1 \
        --beta 0.1 \
        --batch_size 4 \
        --max_length 1024 \
        --max_prompt_length 1024 \
        --tag "$tag" \
        --out "$out_json" || { echo "[FAIL] $tag"; FAILED=1; }
}

# in-task (light-r1)
run_eval dense_lightr1          "$DENSE"                   light-r1
run_eval graft_oracle_lightr1   "$GRAFTS/graft_oracle_a1"  light-r1
run_eval graft_anti_lightr1     "$GRAFTS/graft_anti_a1"    light-r1
run_eval graft_random_lightr1   "$GRAFTS/graft_random42_a1" light-r1
run_eval graft_a05_lightr1      "$GRAFTS/graft_oracle_a05" light-r1
# cross-dataset rider (tulu3)
run_eval dense_tulu3            "$DENSE"                   tulu3
run_eval graft_oracle_tulu3     "$GRAFTS/graft_oracle_a1"  tulu3

echo ""
echo "=== summary ==="
for f in "$OUT"/*.json; do
    python - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ci=d.get("reward_margin_ci95",[float("nan")]*2)
print("%-24s margin=%+.4f  ci=[%+.4f,%+.4f]  acc=%.3f  r_c=%+.3f r_r=%+.3f" % (
    d.get("tag","?"), d.get("reward_margin_mean", float("nan")), ci[0], ci[1],
    d.get("preference_accuracy", float("nan")),
    d.get("reward_chosen_mean", float("nan")),
    d.get("reward_rejected_mean", float("nan"))))
PY
done
echo "done $(date)  FAILED=$FAILED"
exit $FAILED

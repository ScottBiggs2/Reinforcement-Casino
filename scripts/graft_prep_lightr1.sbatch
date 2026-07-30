#!/bin/bash
#SBATCH --job-name=graft_prep
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025
#SBATCH --time=05:00:00
#SBATCH --output=logs/graft_prep_%j.out
#SBATCH --error=logs/graft_prep_%j.err

# P (prep) for the grafting rebuttal experiments (A: necsuff probes, B: margin).
#
# CPU tensor work. Rides an idle v100 via bare --gres=gpu:1 because `short` is
# unschedulable until ~08-03 (2200+ pending, 43% of nodes down). Keep-alive
# sidecar per AGENT_PLAYBOOK 7b: the work is real, the GPU would read idle.
#
# Produces, in order:
#   1. anti-mask (1 - oracle) from the SELF-CONSISTENT src-deltafull oracle
#      (built by 8817043 from the 07-28 retrain -- NOT the orphaned Apr-27 mask
#      whose dense source was deleted; see oracle_sparsity_sweep_lightr1 log)
#   2. matched random masks seeds 42/43/44 under NEW filenames
#      random_matched_lightr1_sp97.5_seed{S}.pt (the old seed42 file covers
#      only 290 tensors, no layer-norms -- mask_composition finding 3)
#   3. four grafted checkpoints for B:
#      graft_oracle_a1 / graft_anti_a1 / graft_random42_a1 / graft_oracle_a0.5
#   4. training_args dump so B can exclude trained pairs on the in-task eval

set -uo pipefail
echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  start $(date)"

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi
cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/xie.yiyi/hf_cache
export OMP_NUM_THREADS=16
export RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu

# Keep-alive: CPU-bound job holding a GPU allocation (playbook 7b).
python -c "import torch,time
gs=[torch.zeros(1024,1024,device=f'cuda:{i}') for i in range(torch.cuda.device_count())]
while True:
    for g in gs: (g@g).sum()
    torch.cuda.synchronize(); time.sleep(45)" &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null || true' EXIT

S=/scratch/xie.yiyi
MASK_DIR=$S/transfer_v1/oracle_masks_llama8b
ORACLE=$MASK_DIR/oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt
ANTI=$MASK_DIR/anti_oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt
DENSE=$S/transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500
BASE=meta-llama/Llama-3.1-8B-Instruct
GRAFTS=$S/graft_ckpts_lightr1
SWEEP_LOG=/home/xie.yiyi/rc-sparse-speed/logs/lr1_mask_sweep_8817043.out
mkdir -p "$GRAFTS"

# ---------------------------------------------------------------------------
# Gate: wait for 8817043 to FINISH the 97.5 mask (file existing != complete;
# torch.save is not atomic). Grep the producing job's own "done" line.
# ---------------------------------------------------------------------------
echo "[gate] waiting for rho=97.5 completion in $SWEEP_LOG"
for i in $(seq 1 240); do
    if grep -q "rho=97.5% done" "$SWEEP_LOG" 2>/dev/null; then
        echo "[gate] ready after ${i} min"; break
    fi
    if [ "$i" -eq 240 ]; then echo "FATAL: 97.5 mask never completed"; exit 1; fi
    sleep 60
done
test -s "$ORACLE" || { echo "FATAL: $ORACLE missing despite done-line"; exit 1; }

set -e

echo "=== [1/4] anti-mask ==="
if [ ! -s "$ANTI" ]; then
    python src/warm_start/anti_mask.py \
        --reference_mask "$ORACLE" --output_file "$ANTI"
else echo "[skip] $ANTI"; fi

echo "=== [2/4] matched random masks (291 tensors incl. layer-norms) ==="
for seed in 42 43 44; do
    out=$MASK_DIR/random_matched_lightr1_sp97.5_seed${seed}.pt
    if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
    python src/warm_start/random_mask_baseline.py \
        --reference_mask "$ORACLE" \
        --sparsity_percent 97.5 \
        --seed "$seed" \
        --output_file "$out" \
        --compare_to_reference
done

echo "=== [3/4] grafted checkpoints ==="
graft () { # name mask mode alpha
    local dir=$GRAFTS/$1
    if [ -s "$dir/graft_meta.json" ]; then echo "[skip] $dir"; return; fi
    python src/evaluation/apply_delta_patch_and_save.py \
        --base_model "$BASE" \
        --finetuned_model "$DENSE" \
        --mask "$2" \
        --patch_mode "$3" \
        --alpha "$4" \
        --output_dir "$dir"
}
graft graft_oracle_a1   "$ORACLE" delta_only      1.0
graft graft_anti_a1     "$ORACLE" anti_delta_only 1.0
graft graft_random42_a1 "$MASK_DIR/random_matched_lightr1_sp97.5_seed42.pt" delta_only 1.0
graft graft_oracle_a05  "$ORACLE" delta_only      0.5

echo "=== [4/4] training_args dump (for B's trained-pair exclusion) ==="
python scripts/read_training_args.py "$DENSE/training_args.bin" \
    | tee "$GRAFTS/dense_training_args.txt" || true

echo "=== inventory ==="
ls -la "$MASK_DIR" | grep -E "matched|anti_.*deltafull"
du -sh "$GRAFTS"/* 2>/dev/null
echo "done $(date)"

export HF_TOKEN="hf_YourTokenHere"

sbatch scripts/run_evals_slurm.sh \
  --model_path "$HOME/rl_casino_train/tulu3_500_h200_fresh_0409_again/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500" \
  --trust_remote_code \
  --output_dir "results/eval_tulu3_500_h200_fresh_0409_again_ckpt500"

  Now we can work with this for several evals here: 

  /scratch/biggs.s/rl_casino_sparse_train/tulu3_500_h200_fresh_0409_again/checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct/fullpipe_sparse_checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct_tulu3_500_h200_fresh_0409_again/final_model

  sbatch scripts/run_evals_slurm.sh \
  --model_path /scratch/biggs.s/rl_casino_sparse_train/tulu3_500_h200_fresh_0409_again/checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct/fullpipe_sparse_checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct_tulu3_500_h200_fresh_0409_again/final_model \
  --trust_remote_code \
  --output_dir "results/eval_tulu3_500_h200_fresh_0409_again_ckpt500"


Sparse speed ablation: 

(base) [biggs.s@explorer-02 rl_casino]$ export H200_BSR_OUT="${SCRATCH_USER_ROOT}/rl_casino_h200_bsr/run_${SLURM_JOB_ID:-manual}"
(base) [biggs.s@explorer-02 rl_casino]$ export SCRATCH_USER_ROOT="/scratch/${USER}"
(base) [biggs.s@explorer-02 rl_casino]$ export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export H200_BSR_STEPS_PER_PHASE=100
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_MAX_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=1024
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
export DPO_GRADIENT_ACCUMULATION_STEPS=64
export HF_DATASETS_CACHE="${SCRATCH_USER_ROOT}/hf_cache/datasets"
(base) [biggs.s@explorer-02 rl_casino]$ unset NUM_STEPS_DPO 2>/dev/null || true
(base) [biggs.s@explorer-02 rl_casino]$ export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_CONSOLE=off
(base) [biggs.s@explorer-02 rl_casino]$ sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
Submitted batch job XXXXXX


(base) [biggs.s@explorer-01 rl_casino]$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5991736       gpu mask_gt_  biggs.s PD       0:00      1 (Priority)
           5991892       gpu verify_g  biggs.s PD       0:00      1 (Priority)
           5988045       gpu h200_bsr  biggs.s  R    1:24:17      1 d4055
(base) [biggs.s@explorer-01 rl_casino]$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5991736       gpu mask_gt_  biggs.s  R       6:27      1 d1029


Job ID | Job Content | Date | Time | Status |
5988045| Sparse ablation | 4/16 | 20:30 | Failed - Stale File Handle again |
5988077| Mask vs GT analysis Again| 4/16 | 20:30| CKA and ER Failed |
5991736| Mask vs GT analysis Again^2| 4/16 | 22:02| failed |
5991892| GRPO verification script 1| 4/16 | 22:04 | failed |
5995131| Sparse speed ablation | 4/16 | 22:40 | not so good but it ran | 
5998766| GRPO verification script 1 again | 23:20 | Winning |
5999931| Mask vs GT again^3 | 4/16 | 23:30 | Failed |
6053065| Mask vs GT again^4| 4/17| 8:45 | Yes results, no plotting? |
6054255| Sparse speed Ablation Again| 4/17 | 9:05 | Ran, partially. Need to record results in table and resue with more sparsity values |
6116134| 
---

### Mask-GT CSV sanity (finite CKA + effective rank before trusting PNGs)

```bash
CSV="/path/to/layer_metrics_gt_vs_....csv"
python3 -c "
import csv, math, sys
def ok(x):
    s = str(x).strip().lower()
    if s in ('', 'nan', 'none'): return False
    try: v=float(s); return math.isfinite(v)
    except Exception: return False
with open(sys.argv[1], newline='') as f: r=list(csv.DictReader(f))
cka=sum(1 for row in r if ok(row.get('cka','')))
er=sum(1 for row in r if ok(row.get('effective_rank_a_norm','')) or ok(row.get('effective_rank_b_norm','')))
print('rows', len(r), 'finite_cka', cka, 'finite_er_norm', er)
" "$CSV"
```

If `finite_cka` is 0: ensure `cka_gt_vs_*.json` exists and `mask_to_cka` succeeded; `git pull` then re-run mask-GT. If `finite_er_norm` is 0: job must not pass `--skip_effective_rank` — use current `run_mask_analysis_vs_ground_truth.sh` (forces ER on unless `MASK_GT_SKIP_EFFECTIVE_RANK=1`).

---

### Mask vs ground truth — full suite (Jaccard + CKA + effective rank + plots)

Use this when **CKA and effective rank are the point** of the run. Submit from **repo root** so `logs/` and `scripts/` resolve. Do **not** set `MASK_GT_SKIP_EFFECTIVE_RANK=1`.

```bash
export REPO="${REPO:-$HOME/rl_casino}"
cd "$REPO" || exit 1
git pull   # ensure latest run_mask_analysis_vs_ground_truth.sh + mask_to_cka + dpo_text_normalize

export MASK_ANALYSIS_DIR="/scratch/biggs.s/rl_casino_masks/tulu3_500_h200_fresh_0409_again"
export GROUND_TRUTH_BASENAME="checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct.pt"

# Full suite (explicit — inherited by sbatch --export=ALL)
export RUN_MASK_CKA=1
export MASK_GT_SKIP_EFFECTIVE_RANK=0
export EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=0

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# export HF_TOKEN="hf_..."   # required if Llama is gated on this cluster

# Optional: more CKA calibration prompts / wider GPU batches (defaults usually fine)
# export CKA_N_SAMPLES=64
# export CKA_BATCH_SIZE=2

# CUDA allocator (set in script too; harmless to export)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs

sbatch --export=ALL \
  scripts/run_mask_analysis_vs_ground_truth.sh

# Note job id from sbatch, then e.g.:
#   tail -f logs/mask_gt_analysis_<JOBID>.out
# Wait until the log shows plot_layer_metrics_csv finishing and lines like
#   [plot_layer_metrics] layer_metrics_....csv: finite_cka=... finite_er_a_norm=...
# before copying PNGs; otherwise you may copy stale images from an earlier run.
# Artifacts:
#   ${MASK_ANALYSIS_DIR}/comparisons_vs_ground_truth/{jaccard_*,cka_*,layer_metrics_*}.csv/json
#   ${MASK_ANALYSIS_DIR}/comparisons_vs_ground_truth/plots/*_plots.png
# Tee log: logs/mask_gt_analysis_gt_analysis_<dir_basename>_<JOBID>.log
```

**Plots only (compute node — do not run `plot_layer_metrics_csv.py` on the login node):** if CSVs exist but PNGs are missing or stale, regenerate plots with a short CPU job:

```bash
export REPO="${REPO:-$HOME/rl_casino}"
cd "$REPO" || exit 1
export MASK_ANALYSIS_DIR="/scratch/biggs.s/rl_casino_masks/tulu3_500_h200_fresh_0409_again"
# Optional: delete old *_plots.png in plots/ before regenerating
# export PLOT_MASK_GT_REMOVE_OLD=1
sbatch scripts/sbatch_plot_mask_gt_comparisons.sh
# tail -f logs/mask_gt_plots_<JOBID>.out
```

**Copy PNGs to home after success** (`PLOT_DIR` must be set — do not rely on an empty env). Only copy after the mask-GT job (or plot-only job above) has finished writing PNGs:

```bash
export MASK_ANALYSIS_DIR="/scratch/biggs.s/rl_casino_masks/tulu3_500_h200_fresh_0409_again"
export PLOT_DIR="${MASK_ANALYSIS_DIR}/comparisons_vs_ground_truth/plots"
FIG_SNAP="${HOME}/figs/mask_gt_plots_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FIG_SNAP"
shopt -s nullglob
PNG=( "$PLOT_DIR"/*.png )
shopt -u nullglob
echo "Found ${#PNG[@]} PNG(s) under $PLOT_DIR"
[ "${#PNG[@]}" -gt 0 ] && cp -av "${PNG[@]}" "$FIG_SNAP"/ && echo "Copied to $FIG_SNAP"
```

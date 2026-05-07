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

# Speed Benchmark Again: 
(base) [biggs.s@explorer-02 rl_casino]$ export RL_CASINO_BSR_GRAD_INPUT_MODE=dense
sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
Submitted batch job 6498127


export GRPO_TARGET_STEPS=800



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

05/05 4pm

# Score gaps again
(base) [biggs.s@explorer-01 rl_casino]$ export CHECKPOINT_DTYPE=bfloat16   # explicit
sbatch scripts/slurm_mask_score_gap_light_r1.slurm
Submitted batch job 6573913

# Make masks before launching interp suite on them later
(base) [biggs.s@explorer-01 rl_casino]$ cd ~/rl_casino
export HF_TOKEN="${HF_TOKEN:?}"
export PIPELINE_RUN_ID=dpo5k_dense_tulu3
export DPO_DATASET_KEY=tulu3
export CHECKPOINT_STEP=500
export TARGET_STEP_DPO=200
export SPARSITY_PERCENT=97.5
export MASK_RUN_ID="tulu3_mag_oracle_rand_${PIPELINE_RUN_ID}"
sbatch scripts/sbatch_make_mag_oracle_random_masks_from_run.sh
Submitted batch job 6573918

# relaunching speed ablation yet again... 
(base) [biggs.s@explorer-01 rl_casino]$ sbatch scripts/h200_speed_ablation_v2.sh
Submitted batch job 6573973
(base) [biggs.s@explorer-01 rl_casino]$ 

# and again... 
(base) [biggs.s@explorer-01 rl_casino]$ sbatch scripts/h200_speed_ablation_v2.sh
Submitted batch job 6574079
(base) [biggs.s@explorer-01 rl_casino]$ 


# interp suite launch on tulu3 oracle, 200, and random
  scripts/sbatch_mask_interpretation_suite.she_%j.err \\"
Submitted batch job 6574787
(base) [biggs.s@explorer-01 rl_casino]$ 

# score comparisons with global and hybrid tau methods: 
(base) [biggs.s@explorer-01 rl_casino]$ export CHECKPOINT_DTYPE=bfloat16 SPARSITY_PERCENT=97.5
export CERT_MIN_LAYER_KEEP_RATIO=0 CERT_MATCH_TIE_BREAK=1
export CERT_GLOBAL_MODE=stream
export CERT_TAU_RULE=global CERT_HYBRID_MIN_LAYER_KEEP_RATIO=0
export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_light_r1/run_global_tau
sbatch scripts/slurm_mask_score_gap_light_r1.slurm
Submitted batch job 6575643
(base) [biggs.s@explorer-01 rl_casino]$ export CHECKPOINT_DTYPE=bfloat16 SPARSITY_PERCENT=97.5
export CERT_MIN_LAYER_KEEP_RATIO=0 CERT_MATCH_TIE_BREAK=1
export CERT_GLOBAL_MODE=stream
export CERT_TAU_RULE=hybrid_global_phase CERT_HYBRID_MIN_LAYER_KEEP_RATIO=0.02
export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_light_r1/run_hybrid_tau
sbatch scripts/slurm_mask_score_gap_light_r1.slurm
Submitted batch job 6575644

# 05/06 Morning! 

(base) [biggs.s@explorer-01 rl_casino]$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           6578617       gpu h200_spe  biggs.s  R    1:26:28      1 d4052
           6575644     short mask_sco  biggs.s  R    6:30:10      1 d0139
           6578960     short mask_int  biggs.s  R      52:55      1 c0621

# resubmitted hybrid tau with lighter hyperparams to finish faster while I develop/launch parallelized version

# OG
(base) [biggs.s@explorer-01 rl_casino]$ sbatch --time=08:00:00 --mem=256G --export=ALL,HISTOGRAM_BINS=512  scripts/slurm_mask_score_gap_light_r1.slurm
Submitted batch job 6589014

# Parallelized
(base) [biggs.s@explorer-02 rl_casino]$ export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_parallel/run1
# (export CKPT500_DIR / DELTA_LOG_DIR / TRAIN_ENV / HF_TOKEN if you don’t match defaults)
bash scripts/submit_mask_score_gap_parallel_light_r1.sh
cache_only job_id=6590158
baseline_shard job_id=6590159 (after cache 6590158)
milestone_shard array job_id=6590160 (tasks 0-3)
merge_shards job_id=6590161 (after milestone array 6590160)
Final artifacts under OUT_DIR after job 6590161 completes.
(base) [biggs.s@explorer-02 rl_casino]$ 

# 11:30 Squeue status: 
(base) [biggs.s@explorer-02 rl_casino]$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           6590319       gpu h200_opt  biggs.s PD       0:00      1 (Priority)
           6589014     short mask_sco  biggs.s PD       0:00      1 (Priority)
           6590158     short mgap_par  biggs.s PD       0:00      1 (Priority)
           6590161     short mgap_par  biggs.s PD       0:00      1 (Dependency)
     6590160_[0-3]     short mgap_par  biggs.s PD       0:00      1 (Dependency)
           6590159     short mgap_par  biggs.s PD       0:00      1 (Dependency)
           6590308     short tulu3_pl  biggs.s PD       0:00      1 (Priority)
(base) [biggs.s@explorer-02 rl_casino]$ 

# again....
(base) [biggs.s@explorer-02 rl_casino]$ export ELEM_MASK="/scratch/biggs.s/rl_casino_masks/orch_lr1_grasp6_6376972/random_elem_meta-llama_Llama-3.1-8B-Instruct_light_r1_sp97.5pct_seed42.pt"
sbatch scripts/h200_sparse_adamw_optstep_microbench.slurm
Submitted batch job 6592410

## SparseAdamW optimizer.step() microbench (H200)

Goal: isolate optimizer kernel memory/speed savings (no forward/backward).

- **Run** (paste on Explorer login — repo root; matches `.out` from job **6593156**):

```bash
cd /scratch/${USER}/rl_casino/RL_Casino_Working_Branch   # or your checkout path
mkdir -p logs

export SCRATCH_USER_ROOT="/scratch/${USER}"

# Element mask (same file as run 6593156)
export ELEM_MASK="/scratch/biggs.s/rl_casino_masks/orch_lr1_grasp6_6376972/random_elem_meta-llama_Llama-3.1-8B-Instruct_light_r1_sp97.5pct_seed42.pt"

# Defaults echoed in logs/h200_optstep_mb_<JOBID>.out — set explicitly if you want an exact rerun
export STEPS=50
export TRIM_FRAC=0.10
export LR=5e-7
export BLOCK_SIZE=32

# Element-only (no block column); equivalent to BLOCK_MASK= RUN_BLOCK=0 in the .out
unset BLOCK_MASK
export RUN_BLOCK=0

sbatch scripts/h200_sparse_adamw_optstep_microbench.slurm
```

- **What to open**: `/scratch/biggs.s/rl_casino_optstep_microbench/<JOBID>/elem/optimizer_step_microbench.md`  
  (replace `biggs.s` with `${USER}` if you did not override `OUT_BASE`; Slurm sets `OUT_BASE=/scratch/$USER/rl_casino_optstep_microbench/$JOBID` by default.)
- **Defaults (designed for fairness + speed)**:
  - steps: 50, trim: 10% (drop first/last 5)
  - dtype: bf16, CUDA sync: on
  - caps: `max_total_numel=25_000_000`, `max_tensors=64`

### Latest result snapshot (elem mask, job 6593156)

- **Mask**: `/scratch/biggs.s/rl_casino_masks/orch_lr1_grasp6_6376972/random_elem_meta-llama_Llama-3.1-8B-Instruct_light_r1_sp97.5pct_seed42.pt`
- **Settings**: bf16, block_size=32, steps=50, trim=10%, CUDA sync=on

#### Timing (trimmed mid-window)

| case | optimizer | mean_ms_mid | p50_ms_mid |
|---|---|---:|---:|
| `dense_elem` | `adamw_torch` | 7.1216 | 7.1201 |
| `dense8bit_elem` | `adamw_8bit` | 14.1032 | 14.0989 |
| `sparse_elem` | `sparse_adamw` | 4.3116 | 4.2679 |

- **Speedups (trimmed mean)**:
  - SparseAdamW vs torch AdamW: **x1.652 faster**
  - SparseAdamW vs AdamW 8-bit: **x3.272 faster**

#### Memory / traffic estimates (subset only; proxy)

Active fraction (subset): **0.02499** (≈2.5% active).

| case | param_MB | grad_MB | adam_state_MB_dense(fp32 m+v) | adam_state_MB_sparse(fp32 m+v) | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050 | 1050 | 4200 | 105 | 1470 |
| `dense8bit_elem` | 1050 | 1050 | 4200 | 105 | 1470 |
| `sparse_elem` | 1050 | 1050 | 4200 | 105 | 1470 |


sbatch --time=04:00:00 --mem=196G --export=ALL,HISTOGRAM_BINS=512 scripts/slurm_mask_score_gap_light_r1.slurm
#!/bin/bash
#SBATCH --job-name=grpo_masks_plot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/grpo_masks_plot_%j.out
#SBATCH --error=logs/grpo_masks_plot_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "========================================"

CUDA_LIB=$(dirname "$(find /usr /opt /lib64 /lib -name "libcuda.so.1" 2>/dev/null | head -n1)")
if [ -n "$CUDA_LIB" ] && [ "$CUDA_LIB" != "." ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[gpu] injected libcuda from: $CUDA_LIB"
else
    echo "[warn] libcuda.so.1 not found on this node"
fi

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PRIMARY="/scratch/xie.yiyi/conda_envs/rl_casino"
CONDA_ENV_FALLBACK="/home/xie.yiyi/.conda/envs/rl_casino"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE_GPU="${FORCE_GPU:-1}"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV:-}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
fi

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    if [ -d "$CONDA_ENV_PRIMARY" ]; then
        conda activate "$CONDA_ENV_PRIMARY"
        PYTHON_BIN="python"
        echo "[env] Activated conda env: $CONDA_ENV_PRIMARY"
    elif [ -d "$CONDA_ENV_FALLBACK" ]; then
        conda activate "$CONDA_ENV_FALLBACK"
        PYTHON_BIN="python"
        echo "[env] Activated conda env: $CONDA_ENV_FALLBACK"
    else
        echo "[warn] Conda env not found at either path; using $PYTHON_BIN"
    fi
else
    echo "[warn] conda.sh not found at $CONDA_SH; using $PYTHON_BIN"
fi

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs masks/grpo_verify

MODEL="${MODEL:-google/gemma-3-270m-it}"
N_SAMPLES="${N_SAMPLES:-64}"
SPARSITY="${SPARSITY:-90.0}"
MASK_DIR="${MASK_DIR:-masks/grpo_verify}"
SKIP_DPO="${SKIP_DPO:-0}"
CPU_THREADS="${CPU_THREADS:-}"

GRPO_SNIP_MASK="$MASK_DIR/snip_grpo.pt"
GRPO_CAV_MASK="$MASK_DIR/cav_grpo.pt"
GRPO_FISHER_MASK="$MASK_DIR/fisher_grpo.pt"

DPO_SNIP_MASK="${DPO_SNIP_MASK:-$MASK_DIR/snip_dpo.pt}"
DPO_CAV_MASK="${DPO_CAV_MASK:-$MASK_DIR/cav_dpo.pt}"
DPO_FISHER_MASK="${DPO_FISHER_MASK:-$MASK_DIR/fisher_dpo.pt}"

echo "[config] MODEL=$MODEL"
echo "[config] N_SAMPLES=$N_SAMPLES"
echo "[config] SPARSITY=$SPARSITY"
echo "[config] MASK_DIR=$MASK_DIR"
echo "[config] SKIP_DPO=$SKIP_DPO"
if [ -n "$CPU_THREADS" ]; then
    export OMP_NUM_THREADS="$CPU_THREADS"
    export MKL_NUM_THREADS="$CPU_THREADS"
    export OPENBLAS_NUM_THREADS="$CPU_THREADS"
    export NUMEXPR_NUM_THREADS="$CPU_THREADS"
    export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
    export BLIS_NUM_THREADS="$CPU_THREADS"
    echo "[config] CPU_THREADS=$CPU_THREADS (thread env vars exported)"
else
    echo "[config] CPU_THREADS=auto"
fi
if [ "$SKIP_DPO" = "1" ]; then
    echo "[config] DPO_SNIP_MASK=$DPO_SNIP_MASK"
    echo "[config] DPO_CAV_MASK=$DPO_CAV_MASK"
    echo "[config] DPO_FISHER_MASK=$DPO_FISHER_MASK"
fi

# ============================================================
# GPU preflight (force CUDA by default)
# ============================================================
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu] nvidia-smi detected"
    nvidia-smi -L || true
else
    echo "[gpu] nvidia-smi not found"
fi

if [ "$FORCE_GPU" = "1" ]; then
    echo "[gpu] FORCE_GPU=1 (will fail if CUDA is unavailable)"
    "$PYTHON_BIN" - <<'PY'
import sys
import torch

ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f"[gpu] torch.cuda.is_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if not ok:
    print("[gpu] ERROR: CUDA is not available in this job. Re-submit on a GPU node or check container/runtime setup.")
    sys.exit(1)
print(f"[gpu] using CUDA device 0: {torch.cuda.get_device_name(0)}")
PY
else
    echo "[gpu] FORCE_GPU=0 (CPU fallback allowed)"
fi

# ============================================================
# Step 1: SNIP — GRPO + DPO
# ============================================================
echo ""
echo "=== [1/6] SNIP GRPO mask ==="
"$PYTHON_BIN" src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method snip \
    --mode grpo \
    --n_samples $N_SAMPLES \
    --sparsity $SPARSITY \
    --output $GRPO_SNIP_MASK

echo ""
echo "=== [2/6] SNIP DPO mask ==="
if [ "$SKIP_DPO" = "1" ]; then
    echo "[skip] Using existing DPO SNIP mask: $DPO_SNIP_MASK"
else
    "$PYTHON_BIN" src/cold_start/inference_mask_finder.py \
        --model_name $MODEL \
        --method snip \
        --mode dpo \
        --n_samples $N_SAMPLES \
        --sparsity $SPARSITY \
        --output $DPO_SNIP_MASK
fi

# ============================================================
# Step 2: CAV — GRPO + DPO
# ============================================================
echo ""
echo "=== [3/6] CAV GRPO mask ==="
"$PYTHON_BIN" src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method cav \
    --mode grpo \
    --n_samples $N_SAMPLES \
    --sparsity $SPARSITY \
    --output $GRPO_CAV_MASK

echo ""
echo "=== [4/6] CAV DPO mask ==="
if [ "$SKIP_DPO" = "1" ]; then
    echo "[skip] Using existing DPO CAV mask: $DPO_CAV_MASK"
else
    "$PYTHON_BIN" src/cold_start/inference_mask_finder.py \
        --model_name $MODEL \
        --method cav \
        --mode dpo \
        --n_samples $N_SAMPLES \
        --sparsity $SPARSITY \
        --output $DPO_CAV_MASK
fi

# ============================================================
# Step 3: Fisher — GRPO + DPO
# ============================================================
echo ""
echo "=== [5/6] Fisher GRPO mask ==="
"$PYTHON_BIN" src/cold_start/cold_mask_finder.py \
    --model_name $MODEL \
    --mode grpo \
    --n_calibration_samples $N_SAMPLES \
    --sparsity_percent $SPARSITY \
    --output_file $GRPO_FISHER_MASK

echo ""
echo "=== [6/6] Fisher DPO mask ==="
if [ "$SKIP_DPO" = "1" ]; then
    echo "[skip] Using existing DPO Fisher mask: $DPO_FISHER_MASK"
else
    "$PYTHON_BIN" src/cold_start/cold_mask_finder.py \
        --model_name $MODEL \
        --mode dpo \
        --n_calibration_samples $N_SAMPLES \
        --sparsity_percent $SPARSITY \
        --output_file $DPO_FISHER_MASK
fi

# ============================================================
# Step 4: export per-layer metrics CSVs (GRPO vs DPO per method)
# ============================================================
echo ""
echo "=== Exporting layer metrics CSVs ==="

if [ -f "$GRPO_SNIP_MASK" ] && [ -f "$DPO_SNIP_MASK" ]; then
    "$PYTHON_BIN" src/cold_start/export_layer_metrics_csv.py \
        $GRPO_SNIP_MASK $DPO_SNIP_MASK \
        --output $MASK_DIR/layer_metrics_snip_grpo_vs_dpo.csv
else
    echo "[warn] Skipping SNIP GRPO-vs-DPO CSV (missing mask file)."
fi

if [ -f "$GRPO_CAV_MASK" ] && [ -f "$DPO_CAV_MASK" ]; then
    "$PYTHON_BIN" src/cold_start/export_layer_metrics_csv.py \
        $GRPO_CAV_MASK $DPO_CAV_MASK \
        --output $MASK_DIR/layer_metrics_cav_grpo_vs_dpo.csv
else
    echo "[warn] Skipping CAV GRPO-vs-DPO CSV (missing mask file)."
fi

if [ -f "$GRPO_FISHER_MASK" ] && [ -f "$DPO_FISHER_MASK" ]; then
    "$PYTHON_BIN" src/cold_start/export_layer_metrics_csv.py \
        $GRPO_FISHER_MASK $DPO_FISHER_MASK \
        --output $MASK_DIR/layer_metrics_fisher_grpo_vs_dpo.csv
else
    echo "[warn] Skipping Fisher GRPO-vs-DPO CSV (missing mask file)."
fi

# Also compare across methods within GRPO (sanity: do the three methods agree?)
"$PYTHON_BIN" src/cold_start/export_layer_metrics_csv.py \
    $GRPO_SNIP_MASK $GRPO_CAV_MASK \
    --output $MASK_DIR/layer_metrics_grpo_snip_vs_cav.csv

"$PYTHON_BIN" src/cold_start/export_layer_metrics_csv.py \
    $GRPO_SNIP_MASK $GRPO_FISHER_MASK \
    --output $MASK_DIR/layer_metrics_grpo_snip_vs_fisher.csv

# ============================================================
# Step 5: plot — individual CSVs (existing behavior)
# ============================================================
echo ""
echo "=== Generating individual plots ==="
"$PYTHON_BIN" src/cold_start/plot_layer_metrics_csv.py \
    --input-dir $MASK_DIR

# ============================================================
# Step 5b: compare plot — SNIP vs CAV overlaid (GRPO vs DPO)
# ============================================================
echo ""
echo "=== Generating SNIP vs CAV compare plot ==="
if [ -f "$MASK_DIR/layer_metrics_snip_grpo_vs_dpo.csv" ] && [ -f "$MASK_DIR/layer_metrics_cav_grpo_vs_dpo.csv" ]; then
    "$PYTHON_BIN" src/cold_start/plot_layer_metrics_csv.py \
        --compare \
        --csv-a $MASK_DIR/layer_metrics_snip_grpo_vs_dpo.csv \
        --csv-b $MASK_DIR/layer_metrics_cav_grpo_vs_dpo.csv \
        --label-a "SNIP" \
        --label-b "CAV" \
        --output $MASK_DIR/compare_SNIP_vs_CAV_grpo_dpo.png
else
    echo "[warn] Skipping SNIP-vs-CAV compare plot (missing CSVs)."
fi

# ============================================================
# Step 6: quick summary — aggregate Jaccard for each pair
# ============================================================
echo ""
echo "=== Aggregate Jaccard Summary ==="
MASK_DIR_BASENAME="$MASK_DIR"
export MASK_DIR_BASENAME GRPO_SNIP_MASK GRPO_CAV_MASK GRPO_FISHER_MASK DPO_SNIP_MASK DPO_CAV_MASK DPO_FISHER_MASK
"$PYTHON_BIN" - <<'EOF'
import torch, os

MASK_DIR = os.environ.get("MASK_DIR_BASENAME", "masks/grpo_verify")

def load(p):
    d = torch.load(p, map_location="cpu")
    return d["masks"] if "masks" in d else d

def bname(env_key, default):
    return os.path.basename(os.environ.get(env_key, default))

pairs = [
    ("SNIP  GRPO vs DPO  ", bname("GRPO_SNIP_MASK", "snip_grpo.pt"), bname("DPO_SNIP_MASK", "snip_dpo.pt")),
    ("CAV   GRPO vs DPO  ", bname("GRPO_CAV_MASK", "cav_grpo.pt"), bname("DPO_CAV_MASK", "cav_dpo.pt")),
    ("Fisher GRPO vs DPO ", bname("GRPO_FISHER_MASK", "fisher_grpo.pt"), bname("DPO_FISHER_MASK", "fisher_dpo.pt")),
    ("GRPO  SNIP vs CAV  ", bname("GRPO_SNIP_MASK", "snip_grpo.pt"), bname("GRPO_CAV_MASK", "cav_grpo.pt")),
    ("GRPO  SNIP vs Fisher", bname("GRPO_SNIP_MASK", "snip_grpo.pt"), bname("GRPO_FISHER_MASK", "fisher_grpo.pt")),
]

for label, fa, fb in pairs:
    pa, pb = os.path.join(MASK_DIR, fa), os.path.join(MASK_DIR, fb)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"  {label}: MISSING")
        continue
    a, b = load(pa), load(pb)
    common = set(a) & set(b)
    inter = sum((a[k].bool() & b[k].bool()).sum().item() for k in common)
    union = sum((a[k].bool() | b[k].bool()).sum().item() for k in common)
    j = inter / union if union > 0 else 0.0
    print(f"  {label}: Jaccard = {j:.4f}")
EOF

echo ""
echo "Plots saved to: $MASK_DIR/*_plots.png"
echo ""
echo "========================================"
echo "Finished : $(date)"
echo "========================================"

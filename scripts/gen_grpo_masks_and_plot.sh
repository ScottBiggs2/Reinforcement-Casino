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

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "========================================"

source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/xie.yiyi/conda_envs/rl_casino

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs masks/grpo_verify

MODEL="meta-llama/Llama-3.1-8B-Instruct"
N_SAMPLES=64
SPARSITY=90.0
MASK_DIR="masks/grpo_verify"

# ============================================================
# Step 1: SNIP — GRPO + DPO
# ============================================================
echo ""
echo "=== [1/6] SNIP GRPO mask ==="
python src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method snip \
    --mode grpo \
    --n_samples $N_SAMPLES \
    --sparsity $SPARSITY \
    --output $MASK_DIR/snip_grpo.pt

echo ""
echo "=== [2/6] SNIP DPO mask ==="
python src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method snip \
    --mode dpo \
    --n_samples $N_SAMPLES \
    --sparsity $SPARSITY \
    --output $MASK_DIR/snip_dpo.pt

# ============================================================
# Step 2: CAV — GRPO + DPO
# ============================================================
echo ""
echo "=== [3/6] CAV GRPO mask ==="
python src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method cav \
    --mode grpo \
    --n_samples $N_SAMPLES \
    --sparsity $SPARSITY \
    --output $MASK_DIR/cav_grpo.pt

echo ""
echo "=== [4/6] CAV DPO mask ==="
python src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method cav \
    --mode dpo \
    --n_samples $N_SAMPLES \
    --sparsity $SPARSITY \
    --output $MASK_DIR/cav_dpo.pt

# ============================================================
# Step 3: Fisher — GRPO + DPO
# ============================================================
echo ""
echo "=== [5/6] Fisher GRPO mask ==="
python src/cold_start/cold_mask_finder.py \
    --model_name $MODEL \
    --mode grpo \
    --n_calibration_samples $N_SAMPLES \
    --sparsity_percent $SPARSITY \
    --output_file $MASK_DIR/fisher_grpo.pt

echo ""
echo "=== [6/6] Fisher DPO mask ==="
python src/cold_start/cold_mask_finder.py \
    --model_name $MODEL \
    --mode dpo \
    --n_calibration_samples $N_SAMPLES \
    --sparsity_percent $SPARSITY \
    --output_file $MASK_DIR/fisher_dpo.pt

# ============================================================
# Step 4: export per-layer metrics CSVs (GRPO vs DPO per method)
# ============================================================
echo ""
echo "=== Exporting layer metrics CSVs ==="

python src/cold_start/export_layer_metrics_csv.py \
    $MASK_DIR/snip_grpo.pt $MASK_DIR/snip_dpo.pt \
    --output $MASK_DIR/layer_metrics_snip_grpo_vs_dpo.csv

python src/cold_start/export_layer_metrics_csv.py \
    $MASK_DIR/cav_grpo.pt $MASK_DIR/cav_dpo.pt \
    --output $MASK_DIR/layer_metrics_cav_grpo_vs_dpo.csv

python src/cold_start/export_layer_metrics_csv.py \
    $MASK_DIR/fisher_grpo.pt $MASK_DIR/fisher_dpo.pt \
    --output $MASK_DIR/layer_metrics_fisher_grpo_vs_dpo.csv

# Also compare across methods within GRPO (sanity: do the three methods agree?)
python src/cold_start/export_layer_metrics_csv.py \
    $MASK_DIR/snip_grpo.pt $MASK_DIR/cav_grpo.pt \
    --output $MASK_DIR/layer_metrics_grpo_snip_vs_cav.csv

python src/cold_start/export_layer_metrics_csv.py \
    $MASK_DIR/snip_grpo.pt $MASK_DIR/fisher_grpo.pt \
    --output $MASK_DIR/layer_metrics_grpo_snip_vs_fisher.csv

# ============================================================
# Step 5: plot — individual CSVs (existing behavior)
# ============================================================
echo ""
echo "=== Generating individual plots ==="
python src/cold_start/plot_layer_metrics_csv.py \
    --input-dir $MASK_DIR

# ============================================================
# Step 5b: compare plot — SNIP vs CAV overlaid (GRPO vs DPO)
# ============================================================
echo ""
echo "=== Generating SNIP vs CAV compare plot ==="
python src/cold_start/plot_layer_metrics_csv.py \
    --compare \
    --csv-a $MASK_DIR/layer_metrics_snip_grpo_vs_dpo.csv \
    --csv-b $MASK_DIR/layer_metrics_cav_grpo_vs_dpo.csv \
    --label-a "SNIP" \
    --label-b "CAV" \
    --output $MASK_DIR/compare_SNIP_vs_CAV_grpo_dpo.png

# ============================================================
# Step 6: quick summary — aggregate Jaccard for each pair
# ============================================================
echo ""
echo "=== Aggregate Jaccard Summary ==="
python - <<'EOF'
import torch, os

MASK_DIR = "masks/grpo_verify"

def load(p):
    d = torch.load(p, map_location="cpu")
    return d["masks"] if "masks" in d else d

pairs = [
    ("SNIP  GRPO vs DPO  ", "snip_grpo.pt",   "snip_dpo.pt"),
    ("CAV   GRPO vs DPO  ", "cav_grpo.pt",    "cav_dpo.pt"),
    ("Fisher GRPO vs DPO ", "fisher_grpo.pt", "fisher_dpo.pt"),
    ("GRPO  SNIP vs CAV  ", "snip_grpo.pt",   "cav_grpo.pt"),
    ("GRPO  SNIP vs Fisher","snip_grpo.pt",   "fisher_grpo.pt"),
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

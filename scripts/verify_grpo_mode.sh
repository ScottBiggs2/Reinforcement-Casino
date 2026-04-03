#!/bin/bash
#SBATCH --job-name=verify_grpo_mode
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/verify_grpo_mode_%j.out
#SBATCH --error=logs/verify_grpo_mode_%j.err

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "========================================"

source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/xie.yiyi/conda_envs/rl_casino

cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs /tmp/grpo_verify

MODEL="google/gemma-3-270m-it"

# ============================================================
# 第一层：data loading — 不需要模型
# ============================================================
echo ""
echo "=== 第一层: Data loader 验证 ==="
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from src.cold_start.inference_mask_finder import load_calibration_samples

print("--- GRPO loader ---")
pos, neg = load_calibration_samples(n_samples=4, mode="grpo")
assert len(pos) == len(neg) == 4, f"期望4对, 实际 {len(pos)}/{len(neg)}"
for i, (p, n) in enumerate(zip(pos, neg)):
    assert len(p) > len(n), f"样本{i}: positive({len(p)}) 应比 negative({len(n)}) 长"
    assert p.startswith(n), f"样本{i}: positive 应以 prompt 开头"
    print(f"  [{i}] neg={len(n):4d}chars  pos={len(p):5d}chars  ✓")

print("--- DPO loader (回归) ---")
chosen, rejected = load_calibration_samples(n_samples=4, mode="dpo")
assert len(chosen) == len(rejected) == 4, f"期望4对, 实际 {len(chosen)}/{len(rejected)}"
print(f"  {len(chosen)} chosen/rejected pairs ✓")

print("第一层 PASSED")
EOF

if [ $? -ne 0 ]; then
    echo "第一层 FAILED — 停止"
    exit 1
fi

# ============================================================
# 第二层：端到端生成 mask，检查 metadata
# ============================================================
echo ""
echo "=== 第二层: 端到端 mask 生成 (SNIP, 8 samples) ==="

python src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method snip \
    --mode grpo \
    --n_samples 8 \
    --sparsity 90.0 \
    --output /tmp/grpo_verify/mask_grpo.pt

python src/cold_start/inference_mask_finder.py \
    --model_name $MODEL \
    --method snip \
    --mode dpo \
    --n_samples 8 \
    --sparsity 90.0 \
    --output /tmp/grpo_verify/mask_dpo.pt

python - <<'EOF'
import torch, sys

for path, label in [("/tmp/grpo_verify/mask_grpo.pt", "GRPO"),
                    ("/tmp/grpo_verify/mask_dpo.pt",  "DPO")]:
    d = torch.load(path, map_location="cpu")
    m = d["metadata"]
    masks = d["masks"]
    total = sum(v.numel() for v in masks.values())
    kept  = sum(v.sum().item() for v in masks.values())
    print(f"[{label}]  mode={m['mode']}  dataset={m['dataset']}")
    print(f"         sparsity={100*(1-kept/total):.1f}%  tensors={len(masks)}")
    assert m["mode"] == label.lower(), f"{label}: metadata mode 不对"

print("第二层 PASSED")
EOF

if [ $? -ne 0 ]; then
    echo "第二层 FAILED — 停止"
    exit 1
fi

# ============================================================
# 第三层：Jaccard 对比 — GRPO mask 应该和 DPO mask 有差异
# ============================================================
echo ""
echo "=== 第三层: Jaccard 对比 (GRPO vs DPO) ==="

python src/cold_start/mask_to_jaccard.py \
    --mask_a /tmp/grpo_verify/mask_dpo.pt \
    --mask_b /tmp/grpo_verify/mask_grpo.pt

python - <<'EOF'
import torch, json

def load_masks(path):
    d = torch.load(path, map_location="cpu")
    return d["masks"] if "masks" in d else d

a = load_masks("/tmp/grpo_verify/mask_dpo.pt")
b = load_masks("/tmp/grpo_verify/mask_grpo.pt")

common = set(a) & set(b)
inter = sum((a[k].bool() & b[k].bool()).sum().item() for k in common)
union = sum((a[k].bool() | b[k].bool()).sum().item() for k in common)
jaccard = inter / union if union > 0 else 0.0
print(f"Aggregate Jaccard (DPO vs GRPO): {jaccard:.4f}")

assert jaccard < 0.999, f"Jaccard={jaccard:.4f} 太高 — 两个 loader 可能返回了相同数据"
assert jaccard > 0.10,  f"Jaccard={jaccard:.4f} 太低 — 样本量太少或 loader 有严重问题"
print(f"Jaccard 在合理范围 (0.10, 0.999) ✓")
print("第三层 PASSED")
EOF

echo ""
echo "========================================"
echo "全部验证通过"
echo "Finished : $(date)"
echo "========================================"

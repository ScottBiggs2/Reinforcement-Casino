#!/usr/bin/env python3
"""
Cold-Start Mask → Jaccard 可读文件
=================================
加载两个 cold start 的 .pt mask 文件，计算 Jaccard 相似度，并写出可读的 JSON 报告。

用法:
  python src/cold_start/mask_to_jaccard.py \
      masks/cold_start_cav_90pct.pt \
      masks/cold_start_snip_90pct.pt \
      --output jaccard_cav_vs_snip.json

  python src/cold_start/mask_to_jaccard.py \
      path/to/mask_a.pt path/to/mask_b.pt
"""

import argparse
import json
import os
import sys

import torch


def load_masks(path):
    """从 .pt 文件加载 masks 字典。"""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], data.get("metadata")
    if isinstance(data, dict):
        return data, None
    raise ValueError(f"无法识别的 mask 格式: {path}")


def compute_jaccard(masks_a, masks_b, device="cpu"):
    """
    计算两个 mask 字典之间的 Jaccard 相似度。
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    per_layer = {}
    total_intersection = 0
    total_union = 0

    common_keys = set(masks_a.keys()) & set(masks_b.keys())
    if not common_keys:
        return {
            "aggregate_jaccard": 0.0,
            "mean_jaccard": 0.0,
            "min_jaccard": 0.0,
            "max_jaccard": 0.0,
            "per_layer": {},
            "note": "两个 mask 没有共同的层名",
        }

    for name in sorted(common_keys):
        a = masks_a[name].to(device).bool()
        b = masks_b[name].to(device).bool()
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        jaccard = inter / union if union > 0 else 0.0
        per_layer[name] = round(jaccard, 6)
        total_intersection += inter
        total_union += union

    aggregate = total_intersection / total_union if total_union > 0 else 0.0
    vals = list(per_layer.values())
    return {
        "aggregate_jaccard": round(aggregate, 6),
        "mean_jaccard": round(sum(vals) / len(vals), 6),
        "min_jaccard": round(min(vals), 6),
        "max_jaccard": round(max(vals), 6),
        "per_layer": per_layer,
        "n_layers": len(per_layer),
        "total_intersection": int(total_intersection),
        "total_union": int(total_union),
    }


def main():
    parser = argparse.ArgumentParser(
        description="将两个 cold start mask 转为可读的 Jaccard JSON 文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mask_a", type=str, help="第一个 mask 文件 (.pt)")
    parser.add_argument("mask_b", type=str, help="第二个 mask 文件 (.pt)")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出 JSON 路径；默认根据两个文件名生成",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="计算设备",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: 未检测到 CUDA，使用 CPU", file=sys.stderr)
        args.device = "cpu"

    if not os.path.isfile(args.mask_a):
        print(f"错误: 文件不存在: {args.mask_a}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.mask_b):
        print(f"错误: 文件不存在: {args.mask_b}", file=sys.stderr)
        sys.exit(1)

    print("加载 mask A:", args.mask_a)
    masks_a, meta_a = load_masks(args.mask_a)
    print("加载 mask B:", args.mask_b)
    masks_b, meta_b = load_masks(args.mask_b)

    jaccard = compute_jaccard(masks_a, masks_b, device=args.device)

    # 可读报告：汇总 + 可选 metadata
    report = {
        "mask_a": os.path.abspath(args.mask_a),
        "mask_b": os.path.abspath(args.mask_b),
        "jaccard": {
            "aggregate": jaccard["aggregate_jaccard"],
            "mean": jaccard["mean_jaccard"],
            "min": jaccard["min_jaccard"],
            "max": jaccard["max_jaccard"],
            "n_layers": jaccard.get("n_layers"),
            "total_intersection": jaccard.get("total_intersection"),
            "total_union": jaccard.get("total_union"),
        },
        "per_layer_jaccard": jaccard["per_layer"],
    }
    if meta_a:
        report["metadata_a"] = meta_a
    if meta_b:
        report["metadata_b"] = meta_b

    if args.output is None:
        base_a = os.path.splitext(os.path.basename(args.mask_a))[0]
        base_b = os.path.splitext(os.path.basename(args.mask_b))[0]
        try:
            out_dir = os.path.commonpath([os.path.abspath(args.mask_a), os.path.abspath(args.mask_b)])
        except ValueError:
            out_dir = "."
        args.output = os.path.join(out_dir, f"jaccard_{base_a}_vs_{base_b}.json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nJaccard 汇总:")
    print(f"  aggregate: {jaccard['aggregate_jaccard']:.4f}")
    print(f"  mean:      {jaccard['mean_jaccard']:.4f}")
    print(f"  min:       {jaccard['min_jaccard']:.4f}")
    print(f"  max:       {jaccard['max_jaccard']:.4f}")
    print(f"\n已写入: {args.output}")


if __name__ == "__main__":
    main()

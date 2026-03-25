import argparse
import json
import math
import os
import sys
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.stateless import functional_call

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mask_utils import (
    create_mask_from_scores_gpu_efficient,
    compute_jaccard_similarity,
    save_masks
)
from src.cold_start.utils.cav_probes import compute_cav_scores, CAVProbeScorer
from src.cold_start.utils.snip_scorer import compute_snip_scores, dpo_style_preference_loss


def choose_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def summarize_layer_score_distribution(scores: Dict[str, torch.Tensor], top_n_preview: int = 8) -> Dict[str, object]:
    """Print per-layer CAV score stats and return an aggregate summary."""
    rows: List[Dict[str, float]] = []
    for name, t in scores.items():
        if t is None or t.numel() == 0:
            continue
        s = t.detach().float().reshape(-1).cpu()
        rows.append(
            {
                "layer": name,
                "mean": float(s.mean().item()),
                "std": float(s.std(unbiased=False).item()),
                "min": float(s.min().item()),
                "max": float(s.max().item()),
                "numel": float(s.numel()),
            }
        )

    rows = sorted(rows, key=lambda x: x["layer"])
    print("\n=== Per-layer score distribution (before masking) ===")
    for r in rows:
        print(
            f"  {r['layer']}: mean={r['mean']:.6e}, std={r['std']:.6e}, "
            f"min={r['min']:.6e}, max={r['max']:.6e}, n={int(r['numel'])}"
        )

    if not rows:
        return {
            "n_layers": 0,
            "mean_std": 0.0,
            "max_std": 0.0,
            "min_std": 0.0,
            "near_zero_std_layers": 0,
            "weak_signal": True,
        }

    stds = [r["std"] for r in rows]
    mins = [r["min"] for r in rows]
    maxs = [r["max"] for r in rows]

    preview = sorted(rows, key=lambda x: x["std"])[: max(1, top_n_preview)]
    print("\nLowest-variance layers (possible CAV degeneracy):")
    for r in preview:
        print(f"  {r['layer']}: std={r['std']:.6e}, range=({r['min']:.6e}, {r['max']:.6e})")

    near_zero_std_layers = sum(1 for s in stds if s <= 1e-10)
    weak_signal = (max(stds) <= 1e-8) or (near_zero_std_layers == len(stds))
    if weak_signal:
        print(
            "\n⚠ CAV score variance is near-zero across layers. "
            "This suggests weak cold-start signal or score collapse before masking."
        )

    return {
        "n_layers": len(rows),
        "mean_std": float(sum(stds) / len(stds)),
        "max_std": float(max(stds)),
        "min_std": float(min(stds)),
        "global_min": float(min(mins)),
        "global_max": float(max(maxs)),
        "near_zero_std_layers": int(near_zero_std_layers),
        "weak_signal": bool(weak_signal),
    }


def maybe_add_score_noise(scores: Dict[str, torch.Tensor], noise_ratio: float) -> Dict[str, torch.Tensor]:
    """Add tiny jitter to break ties when CAV scores are nearly uniform."""
    if noise_ratio <= 0:
        return scores

    out: Dict[str, torch.Tensor] = {}
    print(f"Adding CAV tie-break noise to scores (ratio={noise_ratio:.3e})...")
    for name, t in scores.items():
        s = t.detach().float().cpu()
        scale = max(float(s.abs().max().item()) * noise_ratio, 1e-12)
        out[name] = s + torch.randn_like(s) * scale
    return out


def summarize_mask_sparsity(masks: Dict[str, torch.Tensor], top_n_preview: int = 8) -> Dict[str, object]:
    """Log per-layer sparsity to verify global allocation is non-uniform."""
    rows = []
    for name, m in masks.items():
        mm = m.detach().float().cpu()
        total = int(mm.numel())
        kept = float(mm.sum().item())
        sparsity = 1.0 - (kept / total if total > 0 else 0.0)
        rows.append({"layer": name, "sparsity": float(sparsity), "total": total})

    rows = sorted(rows, key=lambda x: x["layer"])
    print("\n=== Per-layer mask sparsity ===")
    for r in rows:
        print(f"  {r['layer']}: sparsity={r['sparsity']:.6f} (n={r['total']})")

    if not rows:
        return {"n_layers": 0}

    sparsities = [r["sparsity"] for r in rows]
    print(
        f"Sparsity range across layers: min={min(sparsities):.6f}, "
        f"max={max(sparsities):.6f}, span={max(sparsities) - min(sparsities):.6f}"
    )

    flat_preview = sorted(rows, key=lambda x: x["sparsity"])[: max(1, top_n_preview)]
    print("Lowest sparsity layers preview:")
    for r in flat_preview:
        print(f"  {r['layer']}: {r['sparsity']:.6f}")

    return {
        "n_layers": len(rows),
        "min_sparsity": float(min(sparsities)),
        "max_sparsity": float(max(sparsities)),
        "mean_sparsity": float(sum(sparsities) / len(sparsities)),
        "span": float(max(sparsities) - min(sparsities)),
    }


def effective_rank_normalized(mask_tensor: torch.Tensor, eps: float = 1e-12) -> Optional[float]:
    """Entropy effective rank normalized by max rank for a 2D mask tensor."""
    if mask_tensor.ndim < 2:
        return None
    W = mask_tensor.detach().float().reshape(mask_tensor.shape[0], -1).cpu()
    m, n = W.shape
    max_rank = min(m, n)
    if max_rank <= 0:
        return 0.0
    s = torch.linalg.svdvals(W)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum().item()
    erank = float(math.exp(entropy))
    return erank / float(max_rank)


def summarize_effective_rank(masks: Dict[str, torch.Tensor], threshold: float = 0.3) -> Dict[str, object]:
    """Compute and log normalized effective rank for each layer after masking."""
    rows = []
    for name, m in masks.items():
        val = effective_rank_normalized(m)
        if val is None:
            continue
        rows.append({"layer": name, "erank_norm": float(val)})

    rows = sorted(rows, key=lambda x: x["layer"])
    print("\n=== Per-layer normalized effective rank (post-mask) ===")
    for r in rows:
        print(f"  {r['layer']}: erank_norm={r['erank_norm']:.6f}")

    if not rows:
        return {"n_layers": 0, "fraction_above_threshold": 0.0, "threshold": threshold}

    values = [r["erank_norm"] for r in rows]
    above = sum(1 for v in values if v > threshold)
    print(
        f"Effective-rank summary: mean={sum(values)/len(values):.6f}, "
        f"min={min(values):.6f}, max={max(values):.6f}, "
        f"layers_above_{threshold:.2f}={above}/{len(values)}"
    )
    if above < max(1, int(0.5 * len(values))):
        print(
            "⚠ Effective rank remains low in many layers. "
            "Likely cause: degenerate/low-variance CAV scores in cold start."
        )

    return {
        "n_layers": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "threshold": float(threshold),
        "num_above_threshold": int(above),
        "fraction_above_threshold": float(above / len(values)),
    }


def run_cav_warmup_steps(model, dataloader, device: str, steps: int, lr: float) -> None:
    """Run short DPO-style warmup updates before CAV scoring to strengthen signal."""
    if steps <= 0:
        return

    print(f"\nRunning CAV warm-up for {steps} gradient steps (lr={lr})...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    iterator = iter(dataloader)
    losses: List[float] = []

    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
        rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits
        loss = dpo_style_preference_loss(
            chosen_logits=chosen_logits,
            chosen_ids=chosen_ids,
            chosen_mask=chosen_mask,
            rejected_logits=rejected_logits,
            rejected_ids=rejected_ids,
            rejected_mask=rejected_mask,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    if losses:
        print(f"Warm-up complete. Mean warm-up loss: {sum(losses)/len(losses):.6f}")


def map_neuron_scores_to_weight_scores(
    neuron_scores: Dict[str, torch.Tensor],
    extractor,
    model,
    use_weight_abs: bool = False,
) -> Dict[str, torch.Tensor]:
    param_dict = dict(model.named_parameters())
    scores: Dict[str, torch.Tensor] = {}

    module_to_weight = getattr(extractor, "module_to_weight", {}) if extractor is not None else {}

    # Build a suffix index once for robust fallback matching.
    suffix_to_param = {}
    for pname in param_dict.keys():
        if pname.endswith(".weight"):
            suffix_to_param[pname] = pname

    for module_name, score in neuron_scores.items():
        weight_name = module_to_weight.get(module_name)
        if weight_name is None:
            candidate = f"{module_name}.weight"
            if candidate in param_dict:
                weight_name = candidate
            else:
                # Fallback: find unique parameter ending with "<module_name>.weight"
                target_suffix = f"{module_name}.weight"
                matches = [p for p in suffix_to_param.keys() if p.endswith(target_suffix)]
                if len(matches) == 1:
                    weight_name = matches[0]
        if weight_name is None or weight_name not in param_dict:
            continue

        weight = param_dict[weight_name]
        if weight.dim() != 2:
            continue

        # Use per-weight magnitude to avoid constant row/column broadcast scores.
        # Pure broadcasting makes each layer score matrix effectively rank-1,
        # which leads to rank-collapsed binary masks after global thresholding.
        weight_abs = weight.detach().float().abs().cpu()

        flat_score = score.reshape(-1)
        if flat_score.numel() == weight.shape[0]:
            # Neuron scores align with output dimension -> broadcast across columns.
            mapped = flat_score[:, None].expand(weight.shape[0], weight.shape[1]) * weight_abs
        elif flat_score.numel() == weight.shape[1]:
            # Neuron scores align with input dimension (e.g., down_proj input).
            mapped = flat_score[None, :].expand(weight.shape[0], weight.shape[1]) * weight_abs
        else:
            # Shape mismatch can happen for unexpected module outputs.
            continue

        if use_weight_abs:
            scores[weight_name] = mapped.float().cpu()
        else:
            if flat_score.numel() == weight.shape[0]:
                scores[weight_name] = flat_score[:, None].expand(weight.shape[0], weight.shape[1]).float().cpu()
            else:
                scores[weight_name] = flat_score[None, :].expand(weight.shape[0], weight.shape[1]).float().cpu()

    return scores


def _collect_pooled_downproj_activations(
    model,
    dataloader,
    device: str,
    num_batches: int,
    which: str = "chosen",
    mlp_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Collect pooled down_proj input activations directly from DPO batches."""
    assert which in {"chosen", "rejected"}

    acts: Dict[str, List[torch.Tensor]] = {}
    hooks = []
    current_mask = {"value": None}

    for name, module in model.named_modules():
        if not name.endswith("down_proj"):
            continue
        if mlp_only and "mlp" not in name:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        acts[name] = []

        def _make_hook(lname):
            def _hook(mod, inp, out):
                x = inp[0].detach().float()  # [B, T, H]
                mask = current_mask["value"]
                if mask is not None and mask.dim() == 2 and mask.shape[:2] == x.shape[:2]:
                    m = mask.to(x.device).unsqueeze(-1).float()
                    denom = m.sum(dim=1).clamp_min(1.0)
                    pooled = (x * m).sum(dim=1) / denom
                else:
                    pooled = x.mean(dim=1)
                acts[lname].append(pooled.cpu())
            return _hook

        hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break

            ids = batch[f"{which}_input_ids"].to(device)
            mask = batch[f"{which}_attention_mask"].to(device)
            current_mask["value"] = mask
            _ = model(input_ids=ids, attention_mask=mask)
            current_mask["value"] = None

    for h in hooks:
        h.remove()

    return {k: torch.cat(v, dim=0) for k, v in acts.items() if len(v) > 0}


def collect_activation_scores(
    model,
    dataloader,
    device: str,
    num_batches: int,
    mlp_only: bool,
    use_weight_abs: bool = False,
) -> Dict[str, torch.Tensor]:
    """Score neurons by mean |activation| across chosen + rejected sequences."""
    chosen_acts = _collect_pooled_downproj_activations(
        model=model, dataloader=dataloader, device=device, num_batches=num_batches,
        which="chosen", mlp_only=mlp_only,
    )
    rejected_acts = _collect_pooled_downproj_activations(
        model=model, dataloader=dataloader, device=device, num_batches=num_batches,
        which="rejected", mlp_only=mlp_only,
    )

    neuron_scores: Dict[str, torch.Tensor] = {}
    for name in sorted(set(chosen_acts) | set(rejected_acts)):
        parts = []
        if name in chosen_acts:
            parts.append(chosen_acts[name])
        if name in rejected_acts:
            parts.append(rejected_acts[name])
        combined = torch.cat(parts, dim=0)          # [N, D]
        neuron_scores[name] = combined.abs().mean(dim=0)  # [D]

    return map_neuron_scores_to_weight_scores(neuron_scores, extractor=None, model=model, use_weight_abs=use_weight_abs)


def collect_cav_scores(
    model,
    dataloader,
    device: str,
    num_batches: int,
    mlp_only: bool,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    normalize_per_layer: bool = True,
    use_weight_abs: bool = False,
) -> Dict[str, torch.Tensor]:
    positive_acts = _collect_pooled_downproj_activations(
        model=model, dataloader=dataloader, device=device, num_batches=num_batches,
        which="chosen", mlp_only=mlp_only,
    )
    negative_acts = _collect_pooled_downproj_activations(
        model=model, dataloader=dataloader, device=device, num_batches=num_batches,
        which="rejected", mlp_only=mlp_only,
    )

    scorer = CAVProbeScorer()
    neuron_scores = scorer.score(positive_acts, negative_acts, mag_weight=1.0)

    if normalize_per_layer:
        for k, v in list(neuron_scores.items()):
            vv = v.float()
            vmin = vv.min()
            vmax = vv.max()
            if (vmax - vmin) > 1e-12:
                neuron_scores[k] = (vv - vmin) / (vmax - vmin)
            else:
                neuron_scores[k] = torch.zeros_like(vv)

    weight_scores = map_neuron_scores_to_weight_scores(neuron_scores, extractor=None, model=model, use_weight_abs=use_weight_abs)
    if not weight_scores:
        print("[collect_cav_scores] Warning: mapped weight_scores is empty.")
        print(f"  neuron score layers: {len(neuron_scores)}")
        print("  sample neuron keys:", list(neuron_scores.keys())[:5])
    return weight_scores


def collect_cav_scores_with_debug(
    model,
    dataloader,
    device: str,
    num_batches: int,
    mlp_only: bool,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    normalize_per_layer: bool = True,
    use_weight_abs: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
    chosen_acts = _collect_pooled_downproj_activations(
        model=model, dataloader=dataloader, device=device, num_batches=num_batches,
        which="chosen", mlp_only=mlp_only,
    )
    rejected_acts = _collect_pooled_downproj_activations(
        model=model, dataloader=dataloader, device=device, num_batches=num_batches,
        which="rejected", mlp_only=mlp_only,
    )

    # Build labeled_activations: {layer_name: (X [N, D], y [N])}
    layer_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name in sorted(set(chosen_acts) & set(rejected_acts)):
        pos = chosen_acts[name]   # [N_pos, D]
        neg = rejected_acts[name]  # [N_neg, D]
        X = torch.cat([pos, neg], dim=0)
        y = torch.cat([
            torch.ones(pos.shape[0], dtype=torch.long),
            torch.zeros(neg.shape[0], dtype=torch.long),
        ], dim=0)
        layer_data[name] = (X, y)

    neuron_scores = compute_cav_scores(
        labeled_activations=layer_data,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        device=device,
        normalize_per_layer=normalize_per_layer,
    )

    # module_to_weight: strip trailing ".weight" from parameter names
    module_to_weight: Dict[str, str] = {
        pname[:-7]: pname
        for pname in dict(model.named_parameters())
        if pname.endswith(".weight")
    }

    weight_scores = map_neuron_scores_to_weight_scores(
        neuron_scores, extractor=None, model=model, use_weight_abs=use_weight_abs
    )
    return weight_scores, layer_data, module_to_weight


def summarize_top_weight_groups(scores: Dict[str, torch.Tensor], top_n: int) -> List[Dict]:
    rows: List[Dict] = []
    for name, tensor in scores.items():
        t = tensor.float().cpu()
        rows.append(
            {
                "weight_name": name,
                "mean_score": float(t.mean().item()),
                "max_score": float(t.max().item()),
                "numel": int(t.numel()),
            }
        )
    rows.sort(key=lambda x: x["mean_score"], reverse=True)
    return rows[: max(1, top_n)]


def build_cav_group_diagnostics(
    layer_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    module_to_weight: Dict[str, str],
    top_n: int,
) -> Dict[str, List[Dict]]:
    group_metrics: List[Dict] = []
    for module_name, (x, y) in layer_data.items():
        weight_name = module_to_weight.get(module_name)
        if weight_name is None:
            continue

        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        pos = x[y == 1]
        neg = x[y == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue

        pos_mean = pos.abs().mean(dim=0)
        neg_mean = neg.abs().mean(dim=0)
        shared = torch.minimum(pos_mean, neg_mean).mean().item()
        pos_specific = torch.clamp(pos_mean - neg_mean, min=0.0).mean().item()
        neg_specific = torch.clamp(neg_mean - pos_mean, min=0.0).mean().item()

        group_metrics.append(
            {
                "weight_name": weight_name,
                "shared_score": float(shared),
                "task_a_specific_score": float(pos_specific),
                "task_b_specific_score": float(neg_specific),
            }
        )

    def _top(key: str) -> List[Dict]:
        out = sorted(group_metrics, key=lambda x: x[key], reverse=True)
        return out[: max(1, top_n)]

    return {
        "top_shared_groups": _top("shared_score"),
        "top_task_a_specific_groups": _top("task_a_specific_score"),
        "top_task_b_specific_groups": _top("task_b_specific_score"),
    }


def build_layer_selection_masks(all_scores: Dict[str, torch.Tensor], selected_layers: Set[str]) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for name, score in all_scores.items():
        fill = 1.0 if name in selected_layers else 0.0
        masks[name] = torch.full_like(score, fill_value=fill, dtype=torch.float32).cpu()
    return masks


def evaluate_dpo_loss_with_weight_masks(
    model,
    dataloader,
    device: str,
    weight_masks: Optional[Dict[str, torch.Tensor]],
    num_batches: int,
) -> float:
    model.eval()
    param_refs = dict(model.named_parameters())
    total_loss = 0.0
    seen = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            if weight_masks:
                overrides = {}
                for name, mask in weight_masks.items():
                    if name not in param_refs:
                        continue
                    p = param_refs[name]
                    overrides[name] = p * mask.to(device=p.device, dtype=p.dtype)

                chosen_logits = functional_call(
                    model,
                    overrides,
                    (),
                    {"input_ids": chosen_ids, "attention_mask": chosen_mask},
                    strict=False,
                ).logits
                rejected_logits = functional_call(
                    model,
                    overrides,
                    (),
                    {"input_ids": rejected_ids, "attention_mask": rejected_mask},
                    strict=False,
                ).logits
            else:
                chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
                rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

            loss = dpo_style_preference_loss(
                chosen_logits=chosen_logits,
                chosen_ids=chosen_ids,
                chosen_mask=chosen_mask,
                rejected_logits=rejected_logits,
                rejected_ids=rejected_ids,
                rejected_mask=rejected_mask,
            )
            total_loss += float(loss.item())
            seen += 1

    return total_loss / max(1, seen)


def write_debug_reports(
    args,
    model,
    dataloader,
    device: str,
    score_dict: Dict[str, torch.Tensor],
    layer_data: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    module_to_weight: Optional[Dict[str, str]] = None,
) -> None:
    if not args.debug_out_dir:
        return

    os.makedirs(args.debug_out_dir, exist_ok=True)
    report: Dict[str, object] = {
        "method": args.method,
        "debug_eval_batches": args.debug_eval_batches,
        "top_weight_groups_by_mean_score": summarize_top_weight_groups(score_dict, top_n=args.debug_top_groups),
    }

    if args.method == "cav" and layer_data is not None and module_to_weight is not None:
        cav_diag = build_cav_group_diagnostics(
            layer_data=layer_data,
            module_to_weight=module_to_weight,
            top_n=args.debug_top_groups,
        )
        report.update(cav_diag)

        shared_groups = cav_diag["top_shared_groups"]
        task_a_groups = cav_diag["top_task_a_specific_groups"]
        task_b_groups = cav_diag["top_task_b_specific_groups"]

        if shared_groups and task_a_groups and task_b_groups:
            x = shared_groups[0]["weight_name"]
            y = next((g["weight_name"] for g in task_a_groups if g["weight_name"] != x), task_a_groups[0]["weight_name"])
            z = next((g["weight_name"] for g in task_b_groups if g["weight_name"] != x), task_b_groups[0]["weight_name"])

            ablations: List[Tuple[str, Optional[Set[str]]]] = [
                ("dense_unmasked", None),
                ("x_only_shared", {x}),
                ("y_only_task_a_specific", {y}),
                ("x_plus_y", {x, y}),
                ("z_only_task_b_specific", {z}),
                ("x_plus_z", {x, z}),
            ]

            ablation_results: List[Dict] = []
            for name, selected in ablations:
                masks = None if selected is None else build_layer_selection_masks(score_dict, selected_layers=selected)
                value = evaluate_dpo_loss_with_weight_masks(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                    weight_masks=masks,
                    num_batches=args.debug_eval_batches,
                )
                ablation_results.append(
                    {
                        "name": name,
                        "selected_weight_groups": sorted(list(selected)) if selected is not None else "ALL",
                        "avg_dpo_style_loss": float(value),
                    }
                )

            report["representative_groups"] = {"X_shared": x, "Y_task_a_specific": y, "Z_task_b_specific": z}
            report["group_ablation_results"] = ablation_results

    report_file = os.path.join(args.debug_out_dir, "debug_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved debug report to: {report_file}")


def main(args):
    from src.utils.data_utils import concatenated_dpo_collator_fn, load_dpo_dataset

    device = choose_device(args.force_cpu)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.config.use_cache = False

    dataset = load_dpo_dataset(
        dataset_name=args.dataset_name,
        subset_size=args.subset_size,
        split=args.split,
    )

    collator = partial(concatenated_dpo_collator_fn, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    layer_data = None
    module_to_weight = None
    score_variance_summary = None
    mask_sparsity_summary = None
    erank_summary = None

    def _compute_cav_scores_once():
        nonlocal layer_data, module_to_weight
        if args.debug_out_dir:
            s, layer_data_local, module_to_weight_local = collect_cav_scores_with_debug(
                model=model,
                dataloader=dataloader,
                device=device,
                num_batches=args.num_batches,
                mlp_only=args.mlp_only,
                probe_epochs=args.probe_epochs,
                probe_lr=args.probe_lr,
                probe_weight_decay=args.probe_weight_decay,
                normalize_per_layer=not args.no_layer_norm,
                use_weight_abs=args.weight_abs,
            )
            layer_data = layer_data_local
            module_to_weight = module_to_weight_local
            return s

        return collect_cav_scores(
            model=model,
            dataloader=dataloader,
            device=device,
            num_batches=args.num_batches,
            mlp_only=args.mlp_only,
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr,
            probe_weight_decay=args.probe_weight_decay,
            normalize_per_layer=not args.no_layer_norm,
            use_weight_abs=args.weight_abs,
        )

    if args.method == "activation":
        score_dict = collect_activation_scores(
            model=model,
            dataloader=dataloader,
            device=device,
            num_batches=args.num_batches,
            mlp_only=args.mlp_only,
            use_weight_abs=args.weight_abs,
        )
    elif args.method == "cav":
        score_dict = _compute_cav_scores_once()
        score_variance_summary = summarize_layer_score_distribution(score_dict)

        if score_variance_summary.get("weak_signal", False):
            print("\nDetected weak cold-start CAV signal.")
            if args.cav_warmup_steps > 0:
                run_cav_warmup_steps(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                    steps=args.cav_warmup_steps,
                    lr=args.cav_warmup_lr,
                )
                score_dict = _compute_cav_scores_once()
                score_variance_summary = summarize_layer_score_distribution(score_dict)

            if score_variance_summary.get("weak_signal", False) and args.cav_score_noise_ratio > 0:
                score_dict = maybe_add_score_noise(score_dict, noise_ratio=args.cav_score_noise_ratio)
                score_variance_summary = summarize_layer_score_distribution(score_dict)
    elif args.method == "snip":
        score_dict = compute_snip_scores(
            model=model,
            dataloader=dataloader,
            device=device,
            num_batches=args.num_batches,
            mlp_only=args.mlp_only,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    masks = create_mask_from_scores_gpu_efficient(
        score_dict,
        sparsity_percent=args.sparsity_percent,
        device=device,
        add_tie_break_noise=True,
        min_layer_keep_ratio=args.min_layer_keep_ratio,
        local_pool=args.local_pool,
    )

    if args.method == "cav":
        mask_sparsity_summary = summarize_mask_sparsity(masks)
        erank_summary = summarize_effective_rank(masks, threshold=args.erank_target)

    jaccard_results = None
    if args.reference_mask:
        print(f"\nLoading reference mask from: {args.reference_mask}")
        ref = torch.load(args.reference_mask, map_location="cpu")
        reference_masks = ref["masks"]
        jaccard_results = compute_jaccard_similarity(masks, reference_masks)

    metadata = {
        "method": args.method,
        "sparsity_percent": args.sparsity_percent,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "subset_size": args.subset_size,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "mlp_only": args.mlp_only,
        "device": device,
        "min_layer_keep_ratio": args.min_layer_keep_ratio,
    }
    if args.method == "cav":
        metadata["probe_epochs"] = args.probe_epochs
        metadata["probe_lr"] = args.probe_lr
        metadata["probe_weight_decay"] = args.probe_weight_decay
        metadata["cav_warmup_steps"] = args.cav_warmup_steps
        metadata["cav_warmup_lr"] = args.cav_warmup_lr
        metadata["cav_score_noise_ratio"] = args.cav_score_noise_ratio
        if score_variance_summary is not None:
            metadata["score_variance_summary"] = score_variance_summary
        if mask_sparsity_summary is not None:
            metadata["mask_sparsity_summary"] = mask_sparsity_summary
        if erank_summary is not None:
            metadata["effective_rank_summary"] = erank_summary

    if jaccard_results:
        metadata["jaccard_similarity"] = {
            "aggregate": jaccard_results["aggregate_jaccard"],
            "mean": jaccard_results["mean_jaccard"],
            "min": jaccard_results["min_jaccard"],
            "max": jaccard_results["max_jaccard"],
        }

    model_sanitized = sanitize_model_name(args.model_name)
    output_file = args.output_file or f"masks/cold_{args.method}_{model_sanitized}_sparsity{args.sparsity_percent}pct.pt"
    save_masks(masks, output_file, metadata)
    
    if jaccard_results:
        jaccard_file = output_file.replace(".pt", "_jaccard.json")
        with open(jaccard_file, "w") as f:
            json.dump(jaccard_results, f, indent=2)
        print(f"Detailed Jaccard results saved to: {jaccard_file}")
    write_debug_reports(
        args=args,
        model=model,
        dataloader=dataloader,
        device=device,
        score_dict=score_dict,
        layer_data=layer_data,
        module_to_weight=module_to_weight,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference-only cold-start sparse mask finder")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--dataset_name", type=str, default="qihoo360/Light-R1-DPOData")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--method", type=str, choices=["activation", "cav", "snip"], default="cav")
    parser.add_argument("--sparsity_percent", type=float, default=95.0)
    parser.add_argument("--mlp_only", action="store_true", default=True,
                        help="Only score MLP parameters (recommended, consistent with sparse backprop target)")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--reference_mask", type=str, default=None, help="Optional reference mask to compute Jaccard similarity against.")

    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_layer_norm", action="store_true",
                        help="Disable per-layer z-score normalization of CAV scores. "
                             "Without normalization, later layers dominate mask selection.")
    parser.add_argument("--cav_warmup_steps", type=int, default=0,
                        help="Optional number of warm-up DPO gradient steps before CAV scoring when cold-start signal is weak.")
    parser.add_argument("--cav_warmup_lr", type=float, default=1e-6,
                        help="Learning rate for optional CAV warm-up steps.")
    parser.add_argument("--cav_score_noise_ratio", type=float, default=1e-6,
                        help="Fallback tie-break noise ratio added to CAV scores when score variance is near zero.")
    parser.add_argument("--erank_target", type=float, default=0.3,
                        help="Target threshold for normalized effective rank diagnostics.")
    parser.add_argument(
        "--min_layer_keep_ratio",
        type=float,
        default=0.0,
        help="Optional per-layer keep floor ratio for hybrid masking (e.g., 0.05 keeps at least 5%% in each layer, then allocates remaining budget globally).",
    )
    parser.add_argument(
        "--local_pool", action="store_true",
        help=(
            "Use per-layer mask selection instead of global cross-layer ranking. "
            "Default (off): one global threshold across all weights. "
            "With --local_pool: each weight matrix independently keeps its top keep_frac elements, "
            "giving uniform sparsity per layer."
        ),
    )
    parser.add_argument("--weight_abs", action="store_true",
                        help="Weight neuron scores by parameter magnitudes when mapping to weight scores. "
                             "Prevents rank-1 mask collapse; improves effective rank from ~0.002 to ~0.71.")
    parser.add_argument("--debug_out_dir", type=str, default=None, help="Write CAV/subnetwork debug report JSON to this directory.")
    parser.add_argument("--debug_eval_batches", type=int, default=8, help="Number of batches for debug ablation loss evaluation.")
    parser.add_argument("--debug_top_groups", type=int, default=20, help="Top-N groups to keep in debug summaries.")

    args = parser.parse_args()
    main(args)

import argparse
import json
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
from src.cold_start.utils.activation_hooks import FeatureExtractor
from src.cold_start.utils.cav_probes import compute_cav_scores
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


def map_neuron_scores_to_weight_scores(
    neuron_scores: Dict[str, torch.Tensor],
    extractor: FeatureExtractor,
    model,
) -> Dict[str, torch.Tensor]:
    param_dict = dict(model.named_parameters())
    scores: Dict[str, torch.Tensor] = {}

    for module_name, score in neuron_scores.items():
        weight_name = extractor.module_to_weight.get(module_name)
        if weight_name is None or weight_name not in param_dict:
            continue

        weight = param_dict[weight_name]
        if weight.dim() != 2:
            continue

        flat_score = score.reshape(-1)
        if flat_score.numel() != weight.shape[0]:
            # Shape mismatch can happen for unexpected module outputs.
            continue

        row_scores = flat_score[:, None].expand(weight.shape[0], weight.shape[1])
        scores[weight_name] = row_scores.float().cpu()

    return scores


def collect_activation_scores(model, dataloader, device: str, num_batches: int, mlp_only: bool) -> Dict[str, torch.Tensor]:
    extractor = FeatureExtractor(model, mlp_only=mlp_only)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            extractor.collect_activation_stats_start()
            _ = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            _ = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            extractor.collect_stop()

    neuron_scores = extractor.get_activation_scores()
    weight_scores = map_neuron_scores_to_weight_scores(neuron_scores, extractor, model)
    extractor.close()
    return weight_scores


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
) -> Dict[str, torch.Tensor]:
    extractor = FeatureExtractor(model, mlp_only=mlp_only)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # Compute where the response starts (end of prompt) for each sequence.
            # We use the first position where the padding mask is 0 from the right
            # as a proxy. In practice the prompt occupies the first N tokens.
            chosen_len = chosen_mask.sum(dim=1)  # number of real tokens per sample
            rejected_len = rejected_mask.sum(dim=1)

            # Use per-batch majority prompt length as response_start_idx.
            # This is a reasonable approximation when prompt lengths are similar.
            chosen_resp_start = int(chosen_len.min().item() // 2)  # heuristic: assume ~half is prompt
            rejected_resp_start = int(rejected_len.min().item() // 2)

            extractor.collect_labeled_start(label=1, response_start_idx=chosen_resp_start)
            _ = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            extractor.collect_stop()

            extractor.collect_labeled_start(label=0, response_start_idx=rejected_resp_start)
            _ = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            extractor.collect_stop()

    layer_data = extractor.get_labeled_activations()
    neuron_scores = compute_cav_scores(
        labeled_activations=layer_data,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        device=device,
        normalize_per_layer=normalize_per_layer,
    )

    weight_scores = map_neuron_scores_to_weight_scores(neuron_scores, extractor, model)
    extractor.close()
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
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
    extractor = FeatureExtractor(model, mlp_only=mlp_only)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_resp_start = int(chosen_mask.sum(dim=1).min().item() // 2)
            rejected_resp_start = int(rejected_mask.sum(dim=1).min().item() // 2)

            extractor.collect_labeled_start(label=1, response_start_idx=chosen_resp_start)
            _ = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            extractor.collect_stop()

            extractor.collect_labeled_start(label=0, response_start_idx=rejected_resp_start)
            _ = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            extractor.collect_stop()

    layer_data = extractor.get_labeled_activations()
    neuron_scores = compute_cav_scores(
        labeled_activations=layer_data,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        device=device,
        normalize_per_layer=normalize_per_layer,
    )

    weight_scores = map_neuron_scores_to_weight_scores(neuron_scores, extractor, model)
    module_to_weight = dict(extractor.module_to_weight)
    extractor.close()
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
    from src.utils.data_utils import dpo_collator_fn, load_dpo_dataset

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

    collator = partial(dpo_collator_fn, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    layer_data = None
    module_to_weight = None

    if args.method == "activation":
        score_dict = collect_activation_scores(
            model=model,
            dataloader=dataloader,
            device=device,
            num_batches=args.num_batches,
            mlp_only=args.mlp_only,
        )
    elif args.method == "cav":
        if args.debug_out_dir:
            score_dict, layer_data, module_to_weight = collect_cav_scores_with_debug(
                model=model,
                dataloader=dataloader,
                device=device,
                num_batches=args.num_batches,
                mlp_only=args.mlp_only,
                probe_epochs=args.probe_epochs,
                probe_lr=args.probe_lr,
                probe_weight_decay=args.probe_weight_decay,
                normalize_per_layer=not args.no_layer_norm,
            )
        else:
            score_dict = collect_cav_scores(
                model=model,
                dataloader=dataloader,
                device=device,
                num_batches=args.num_batches,
                mlp_only=args.mlp_only,
                probe_epochs=args.probe_epochs,
                probe_lr=args.probe_lr,
                probe_weight_decay=args.probe_weight_decay,
                normalize_per_layer=not args.no_layer_norm,
            )
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

    masks = create_mask_from_scores_gpu_efficient(score_dict, sparsity_percent=args.sparsity_percent, device=device)

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
    }
    if args.method == "cav":
        metadata["probe_epochs"] = args.probe_epochs
        metadata["probe_lr"] = args.probe_lr
        metadata["probe_weight_decay"] = args.probe_weight_decay

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
    parser.add_argument("--debug_out_dir", type=str, default=None, help="Write CAV/subnetwork debug report JSON to this directory.")
    parser.add_argument("--debug_eval_batches", type=int, default=8, help="Number of batches for debug ablation loss evaluation.")
    parser.add_argument("--debug_top_groups", type=int, default=20, help="Top-N groups to keep in debug summaries.")

    args = parser.parse_args()
    main(args)

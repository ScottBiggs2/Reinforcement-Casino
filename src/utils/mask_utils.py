import torch
import os
import json
from typing import Any, Dict

# Default masking mode:
# - global ranking across all scored weights
# - plus a small per-tensor keep floor to reduce collapse under high sparsity
DEFAULT_MIN_LAYER_KEEP_RATIO = 0.0025
DEFAULT_CHUNKED_SELECTOR_MIN_NUMEL = 250_000_000
DEFAULT_SELECTOR_CHUNK_NUMEL = 25_000_000
DEFAULT_SELECTOR_HIST_BINS = 2048
DEFAULT_SELECTOR_MAX_REFINEMENT_PASSES = 12
DEFAULT_SELECTOR_MAX_CANDIDATES = 8_000_000

# Global top-k over ~5B+ scores exceeds 2^32 elements; CUDA topk (gatherTopK) can assert
# on such 1D tensors. Use CPU top-k above this threshold unless RL_CASINO_MASK_TOPK_ALLOW_GPU=1.
_CUDA_TOPK_SAFE_NUMEL = 2_000_000_000


def pooling_metadata(
    *,
    local_pool: bool,
    min_layer_keep_ratio: float = DEFAULT_MIN_LAYER_KEEP_RATIO,
) -> Dict[str, Any]:
    """
    Fields to merge into saved mask metadata so runs are reproducible and comparable.

    - pooling_mode ``global``: one ranking across all scored weight elements.
    - pooling_mode ``global_with_layer_floor``: global top-k after reserving a per-layer
      keep floor (the default when ``min_layer_keep_ratio`` is left unchanged).
    - pooling_mode ``local_per_tensor``: each 2D weight matrix keeps keep_frac of its
      own elements independently (``--local_pool``).

    Random baselines: ``generate_random_mask.py`` uses a single global threshold on
    concatenated scores; ``random_mask_baseline.py`` uses
    ``create_mask_from_scores_gpu_efficient``.

    For very large models, materializing ``torch.cat`` over all scores may be expensive;
    future work: threshold search with per-chunk counts (exact global sparsity) or
    oversampled per-block top-k then a final global top-k (approximate unless oversample
    is large enough to include all true global top-k).
    """
    if local_pool:
        return {"pooling_mode": "local_per_tensor", "local_pool": True}
    if min_layer_keep_ratio > 0:
        return {
            "pooling_mode": "global_with_layer_floor",
            "local_pool": False,
            "min_layer_keep_ratio": float(min_layer_keep_ratio),
        }
    return {"pooling_mode": "global", "local_pool": False}


def _topk_indices_safe(scores_1d: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """Top-k indices; CPU fallback for very large tensors to avoid CUDA gatherTopK failures."""
    n = scores_1d.numel()
    k = max(0, min(int(k), n))
    if k == 0:
        return scores_1d.new_empty((0,), dtype=torch.long)
    allow_gpu = os.environ.get("RL_CASINO_MASK_TOPK_ALLOW_GPU", "").lower() in ("1", "true", "yes")
    force_cpu = os.environ.get("RL_CASINO_MASK_TOPK_CPU", "").lower() in ("1", "true", "yes")
    use_cpu = scores_1d.is_cuda and (force_cpu or (n > _CUDA_TOPK_SAFE_NUMEL and not allow_gpu))
    if use_cpu:
        print(
            f"  Note: top-k on CPU ({n:,} scores, k={k:,}) — "
            "CUDA top-k can fail when the score vector is extremely large."
        )
        idx = torch.topk(scores_1d.detach().cpu(), k=k, largest=largest, sorted=False).indices
        return idx.to(scores_1d.device, non_blocking=False)
    return torch.topk(scores_1d, k=k, largest=largest, sorted=False).indices


def _create_mask_local(scores_dict, sparsity_percent, device, add_tie_break_noise, tie_break_noise_scale):
    """Per-layer top-k: each weight matrix independently keeps keep_frac of its elements."""
    keep_frac = 1.0 - sparsity_percent / 100.0
    print(f"\n=== Creating Per-layer Local Masks (target sparsity: {sparsity_percent}%) ===")

    masks = {}
    total_params = 0
    total_kept = 0

    for name, score in scores_dict.items():
        if score is None or score.numel() == 0:
            masks[name] = torch.zeros_like(score, dtype=torch.float32).cpu()
            continue

        s = score.to(device=device, dtype=torch.float32)
        n = s.numel()
        n_keep = max(1, int(keep_frac * n))

        flat = s.reshape(-1)
        if add_tie_break_noise:
            scale = max(flat.abs().max().item() * tie_break_noise_scale, 1e-12)
            flat = flat + torch.randn_like(flat) * scale

        idx = _topk_indices_safe(flat, k=n_keep, largest=True)
        mask_flat = torch.zeros(n, device=device, dtype=torch.float32)
        mask_flat[idx] = 1.0

        masks[name] = mask_flat.reshape(score.shape).cpu()
        total_params += n
        total_kept += n_keep

    actual_sparsity = 100.0 - (total_kept / max(total_params, 1) * 100.0)
    print(f"Total parameters: {total_params:,}")
    print(f"Actual keep: {total_kept:,} | Actual sparsity: {actual_sparsity:.4f}%")
    return masks


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _flatten_cpu_chunks(flat: torch.Tensor, chunk_numel: int):
    n = flat.numel()
    for start in range(0, n, chunk_numel):
        end = min(n, start + chunk_numel)
        yield start, end, flat[start:end]


def _compute_noise_scale(valid_scores: Dict[str, torch.Tensor], tie_break_noise_scale: float) -> float:
    max_abs = 0.0
    for score in valid_scores.values():
        local_max = float(score.abs().max().item())
        if local_max > max_abs:
            max_abs = local_max
    return max(max_abs * tie_break_noise_scale, 1e-12)


def _apply_tie_break_noise_inplace(
    valid_scores: Dict[str, torch.Tensor],
    tie_break_noise_scale: float,
) -> None:
    scale = _compute_noise_scale(valid_scores, tie_break_noise_scale)
    for name, score in valid_scores.items():
        noise = torch.randn_like(score) * scale
        valid_scores[name] = score + noise
    print(f"Applied tie-break noise (scale={scale:.3e})")


def _select_floor_indices(
    valid_scores: Dict[str, torch.Tensor],
    min_layer_keep_ratio: float,
    keep_count: int,
):
    floor_indices = {}
    floor_selected = 0
    if min_layer_keep_ratio <= 0:
        for name in valid_scores.keys():
            floor_indices[name] = torch.empty(0, dtype=torch.long)
        return floor_indices, floor_selected

    min_layer_keep_ratio = float(max(0.0, min(1.0, min_layer_keep_ratio)))
    layer_floors = []
    for name, score in valid_scores.items():
        layer_n = score.numel()
        local_floor = int(min_layer_keep_ratio * layer_n)
        local_floor = max(0, min(local_floor, layer_n))
        layer_floors.append((name, local_floor, layer_n))

    requested_floor_total = sum(f for _, f, _ in layer_floors)
    if requested_floor_total > keep_count:
        print(
            f"⚠ Requested per-layer floor keeps {requested_floor_total:,} params, "
            f"but global keep budget is {keep_count:,}. Scaling floors down proportionally."
        )
        scale = keep_count / max(1, requested_floor_total)
        layer_floors = [
            (name, max(0, min(layer_n, int(local_floor * scale))), layer_n)
            for name, local_floor, layer_n in layer_floors
        ]

    for name, local_floor, _layer_n in layer_floors:
        if local_floor <= 0:
            floor_indices[name] = torch.empty(0, dtype=torch.long)
            continue
        flat = valid_scores[name].reshape(-1)
        idx = _topk_indices_safe(flat, k=local_floor, largest=True).detach().cpu()
        floor_indices[name] = torch.sort(idx).values
        floor_selected += int(local_floor)

    print(f"Per-layer floor selected: {floor_selected:,} parameters")
    return floor_indices, floor_selected


def _count_scores_above_threshold(
    valid_scores: Dict[str, torch.Tensor],
    floor_indices: Dict[str, torch.Tensor],
    threshold: float,
    chunk_numel: int,
) -> int:
    total = 0
    for name, score in valid_scores.items():
        flat = score.reshape(-1)
        floor_idx = floor_indices[name]
        for start, end, chunk in _flatten_cpu_chunks(flat, chunk_numel):
            if floor_idx.numel() > 0:
                left = torch.searchsorted(floor_idx, start)
                right = torch.searchsorted(floor_idx, end)
                if right > left:
                    local_exclude = floor_idx[left:right] - start
                    available = torch.ones(chunk.numel(), dtype=torch.bool)
                    available[local_exclude] = False
                    total += int((chunk[available] > threshold).sum().item())
                    continue
            total += int((chunk > threshold).sum().item())
    return total


def _refine_threshold_interval(
    valid_scores: Dict[str, torch.Tensor],
    floor_indices: Dict[str, torch.Tensor],
    remaining_keep: int,
    chunk_numel: int,
):
    finite_min = None
    finite_max = None
    for name, score in valid_scores.items():
        flat = score.reshape(-1)
        floor_idx = floor_indices[name]
        for start, end, chunk in _flatten_cpu_chunks(flat, chunk_numel):
            if floor_idx.numel() > 0:
                left = torch.searchsorted(floor_idx, start)
                right = torch.searchsorted(floor_idx, end)
                if right > left:
                    local_exclude = floor_idx[left:right] - start
                    available = torch.ones(chunk.numel(), dtype=torch.bool)
                    available[local_exclude] = False
                    chunk = chunk[available]
            if chunk.numel() == 0:
                continue
            local_min = float(chunk.min().item())
            local_max = float(chunk.max().item())
            finite_min = local_min if finite_min is None else min(finite_min, local_min)
            finite_max = local_max if finite_max is None else max(finite_max, local_max)

    if finite_min is None or finite_max is None:
        return None

    hist_bins = _get_env_int("RL_CASINO_SELECTOR_HIST_BINS", DEFAULT_SELECTOR_HIST_BINS)
    max_passes = _get_env_int(
        "RL_CASINO_SELECTOR_MAX_REFINEMENT_PASSES",
        DEFAULT_SELECTOR_MAX_REFINEMENT_PASSES,
    )
    candidate_limit = _get_env_int(
        "RL_CASINO_SELECTOR_MAX_CANDIDATES",
        DEFAULT_SELECTOR_MAX_CANDIDATES,
    )

    lo = finite_min
    hi = finite_max
    count_above = 0
    candidate_count = None

    for _ in range(max_passes):
        if hi <= lo:
            break
        hist = torch.zeros(hist_bins, dtype=torch.float64)
        span = hi - lo
        for name, score in valid_scores.items():
            flat = score.reshape(-1)
            floor_idx = floor_indices[name]
            for start, end, chunk in _flatten_cpu_chunks(flat, chunk_numel):
                if floor_idx.numel() > 0:
                    left = torch.searchsorted(floor_idx, start)
                    right = torch.searchsorted(floor_idx, end)
                    if right > left:
                        local_exclude = floor_idx[left:right] - start
                        available = torch.ones(chunk.numel(), dtype=torch.bool)
                        available[local_exclude] = False
                        chunk = chunk[available]
                if chunk.numel() == 0:
                    continue
                in_range = (chunk >= lo) & (chunk <= hi)
                if not torch.any(in_range):
                    continue
                hist += torch.histc(chunk[in_range], bins=hist_bins, min=lo, max=hi).to(torch.float64)

        target_rank = remaining_keep - count_above
        if target_rank <= 0:
            break

        running = 0
        selected_bin = hist_bins - 1
        for bin_idx in range(hist_bins - 1, -1, -1):
            bin_count = int(hist[bin_idx].item())
            if running + bin_count >= target_rank:
                selected_bin = bin_idx
                candidate_count = bin_count
                count_above += running
                break
            running += bin_count

        bin_width = span / hist_bins
        new_lo = lo + selected_bin * bin_width
        new_hi = hi if selected_bin == hist_bins - 1 else lo + (selected_bin + 1) * bin_width

        if candidate_count is not None and candidate_count <= candidate_limit:
            lo, hi = new_lo, new_hi
            break

        lo, hi = new_lo, new_hi

    return lo, hi


def _create_mask_global_chunked(
    valid_scores: Dict[str, torch.Tensor],
    sparsity_percent: float,
    add_tie_break_noise: bool,
    tie_break_noise_scale: float,
    min_layer_keep_ratio: float,
):
    print("Mask selection backend: exact chunked global selector")
    total_params = sum(score.numel() for score in valid_scores.values())
    keep_percent = 100.0 - sparsity_percent
    keep_count = max(1, min(total_params, int(keep_percent / 100.0 * total_params)))

    print(f"Total parameters: {total_params:,}")
    print(f"Target keep count: {keep_count:,} ({keep_percent:.2f}%)")
    if min_layer_keep_ratio > 0:
        print(f"Using hybrid global mask with per-layer keep floor ratio={min_layer_keep_ratio:.4f}")

    if add_tie_break_noise:
        _apply_tie_break_noise_inplace(valid_scores, tie_break_noise_scale)

    floor_indices, floor_selected = _select_floor_indices(
        valid_scores,
        min_layer_keep_ratio=min_layer_keep_ratio,
        keep_count=keep_count,
    )

    remaining_keep = keep_count - floor_selected
    masks_bool = {
        name: torch.zeros_like(score, dtype=torch.bool, device="cpu")
        for name, score in valid_scores.items()
    }
    for name, idx in floor_indices.items():
        if idx.numel() == 0:
            continue
        masks_bool[name].view(-1)[idx] = True

    if remaining_keep > 0:
        chunk_numel = _get_env_int(
            "RL_CASINO_SELECTOR_CHUNK_NUMEL",
            DEFAULT_SELECTOR_CHUNK_NUMEL,
        )
        interval = _refine_threshold_interval(
            valid_scores,
            floor_indices,
            remaining_keep=remaining_keep,
            chunk_numel=chunk_numel,
        )
        if interval is None:
            raise ValueError("Chunked selector found no available scores after floor reservation.")

        lo, hi = interval
        boundary_values = []
        boundary_refs = []
        definitely_selected = 0

        for name, score in valid_scores.items():
            flat = score.reshape(-1)
            floor_idx = floor_indices[name]
            above_parts = []
            boundary_parts = []
            for start, end, chunk in _flatten_cpu_chunks(flat, chunk_numel):
                available = torch.ones(chunk.numel(), dtype=torch.bool)
                if floor_idx.numel() > 0:
                    left = torch.searchsorted(floor_idx, start)
                    right = torch.searchsorted(floor_idx, end)
                    if right > left:
                        local_exclude = floor_idx[left:right] - start
                        available[local_exclude] = False
                if not torch.any(available):
                    continue
                available_chunk = chunk[available]
                available_idx = torch.nonzero(available, as_tuple=False).squeeze(-1) + start
                above_mask = available_chunk > hi
                if torch.any(above_mask):
                    above_parts.append(available_idx[above_mask])
                boundary_mask = (available_chunk >= lo) & (available_chunk <= hi)
                if torch.any(boundary_mask):
                    boundary_idx = available_idx[boundary_mask]
                    boundary_parts.append((boundary_idx, available_chunk[boundary_mask]))

            if above_parts:
                above_idx = torch.cat(above_parts)
                masks_bool[name].view(-1)[above_idx] = True
                definitely_selected += int(above_idx.numel())

            for idx_chunk, val_chunk in boundary_parts:
                boundary_values.append(val_chunk)
                boundary_refs.append((name, idx_chunk))

        needed_from_boundary = remaining_keep - definitely_selected
        if needed_from_boundary < 0:
            raise ValueError(
                "Chunked selector selected more values above the refined threshold than the remaining budget allows."
            )

        if needed_from_boundary > 0:
            if not boundary_values:
                raise ValueError("Chunked selector could not locate boundary candidates for the remaining budget.")
            concat_values = torch.cat(boundary_values)
            if needed_from_boundary > concat_values.numel():
                raise ValueError(
                    f"Chunked selector only found {concat_values.numel()} boundary candidates for "
                    f"{needed_from_boundary} required positions."
                )
            selected = _topk_indices_safe(concat_values, k=needed_from_boundary, largest=True).cpu()
            concat_names = []
            concat_indices = []
            for name, idx_chunk in boundary_refs:
                concat_names.extend([name] * idx_chunk.numel())
                concat_indices.append(idx_chunk)
            concat_indices = torch.cat(concat_indices)
            for sel in selected.tolist():
                masks_bool[concat_names[sel]].view(-1)[int(concat_indices[sel].item())] = True

    masks = {}
    total_kept = 0
    print("Applying global mask to layers...")
    for idx, (name, mask) in enumerate(masks_bool.items()):
        if idx % 50 == 0:
            print(f"  Processing layer {idx+1}/{len(masks_bool)}")
        total_kept += int(mask.sum().item())
        masks[name] = mask.to(dtype=torch.float32)

    actual_sparsity = 100.0 - (total_kept / total_params * 100.0)
    print("\nVerification:")
    print(f"  Target keep: {keep_count:,} ({keep_percent:.2f}%)")
    print(f"  Actual keep: {int(total_kept):,} ({100.0 - actual_sparsity:.2f}%)")
    print(f"  Actual sparsity: {actual_sparsity:.4f}% (target: {sparsity_percent}%)")
    print(f"  Error: {abs(actual_sparsity - sparsity_percent):.6f}%")
    return masks


def _create_mask_global_flat(
    valid_scores: Dict[str, torch.Tensor],
    sparsity_percent: float,
    add_tie_break_noise: bool,
    tie_break_noise_scale: float,
    min_layer_keep_ratio: float,
):
    keep_percent = 100.0 - sparsity_percent
    total_params = sum(score.numel() for score in valid_scores.values())
    keep_count = max(1, int(keep_percent / 100.0 * total_params))
    keep_count = min(keep_count, total_params)

    print(f"Total parameters: {total_params:,}")
    print(f"Target keep count: {keep_count:,} ({keep_percent:.2f}%)")
    if min_layer_keep_ratio > 0:
        print(f"Using hybrid global mask with per-layer keep floor ratio={min_layer_keep_ratio:.4f}")

    # Flatten all scores globally.
    offsets = []
    flat_chunks = []
    cursor = 0
    for name, s in valid_scores.items():
        n = s.numel()
        offsets.append((name, cursor, cursor + n, s.shape))
        flat_chunks.append(s.reshape(-1))
        cursor += n

    all_scores = torch.cat(flat_chunks, dim=0)
    torch.nan_to_num_(all_scores, nan=0.0, posinf=0.0, neginf=0.0)

    if add_tie_break_noise:
        scale = max(all_scores.abs().max().item() * tie_break_noise_scale, 1e-12)
        all_scores = all_scores + torch.randn_like(all_scores) * scale
        print(f"Applied tie-break noise (scale={scale:.3e})")

    global_mask_flat = torch.zeros_like(all_scores, dtype=torch.bool)

    floor_selected = 0
    if min_layer_keep_ratio > 0:
        min_layer_keep_ratio = float(max(0.0, min(1.0, min_layer_keep_ratio)))
        layer_floors = []
        for name, start, end, _shape in offsets:
            layer_n = end - start
            local_floor = int(min_layer_keep_ratio * layer_n)
            local_floor = max(0, min(local_floor, layer_n))
            layer_floors.append((name, start, end, local_floor))

        requested_floor_total = sum(f for _, _, _, f in layer_floors)
        if requested_floor_total > keep_count:
            print(
                f"⚠ Requested per-layer floor keeps {requested_floor_total:,} params, "
                f"but global keep budget is {keep_count:,}. Scaling floors down proportionally."
            )
            scale = keep_count / max(1, requested_floor_total)
            scaled_floors = []
            for name, start, end, f in layer_floors:
                layer_n = end - start
                nf = int(f * scale)
                nf = max(0, min(nf, layer_n))
                scaled_floors.append((name, start, end, nf))
            layer_floors = scaled_floors

        for name, start, end, local_floor in layer_floors:
            if local_floor <= 0:
                continue
            layer_scores = all_scores[start:end]
            local_idx = _topk_indices_safe(layer_scores, k=local_floor, largest=True)
            global_mask_flat[start:end][local_idx] = True
            floor_selected += int(local_floor)

        print(f"Per-layer floor selected: {floor_selected:,} parameters")

    remaining = keep_count - floor_selected
    if remaining > 0:
        all_scores[global_mask_flat] = float("-inf")
        keep_indices = _topk_indices_safe(all_scores, k=remaining, largest=True)
        global_mask_flat[keep_indices] = True

    masks = {}
    total_kept = 0
    print("Applying global mask to layers...")
    for idx, (name, start, end, shape) in enumerate(offsets):
        if idx % 50 == 0:
            print(f"  Processing layer {idx+1}/{len(offsets)}")
        m = global_mask_flat[start:end].reshape(shape)
        total_kept += int(m.sum().item())
        masks[name] = m.to(dtype=torch.float32).cpu()

    actual_sparsity = 100.0 - (total_kept / total_params * 100.0)
    print("\nVerification:")
    print(f"  Target keep: {keep_count:,} ({keep_percent:.2f}%)")
    print(f"  Actual keep: {int(total_kept):,} ({100.0 - actual_sparsity:.2f}%)")
    print(f"  Actual sparsity: {actual_sparsity:.4f}% (target: {sparsity_percent}%)")
    print(f"  Error: {abs(actual_sparsity - sparsity_percent):.6f}%")
    return masks


def create_mask_from_scores_gpu_efficient(
    scores_dict,
    sparsity_percent,
    device='cuda',
    add_tie_break_noise: bool = True,
    tie_break_noise_scale: float = 1e-10,
    min_layer_keep_ratio: float = DEFAULT_MIN_LAYER_KEEP_RATIO,
    local_pool: bool = False,
):
    """
    Create sparse masks from score tensors.

        Default (local_pool=False): *global* ranking with a small per-tensor keep floor,
        so high-scoring layers still compete globally but each scored tensor keeps at least
        a small fraction of weights unless the global budget is too small.

        local_pool=True: *per-layer* ranking — each weight matrix independently
        keeps its top keep_frac elements, giving uniform sparsity per layer.

        Optional hybrid mode (local_pool=False only):
            - If min_layer_keep_ratio > 0, each non-empty layer keeps at least
                floor(min_layer_keep_ratio * layer_numel) parameters.
            - Remaining budget is allocated by global top-k over the full model.
            - Pass min_layer_keep_ratio=0.0 for pure global selection with no floor.
    """
    if local_pool:
        print("Mask pooling: local (each weight matrix ranked independently; use --local_pool)")
        return _create_mask_local(scores_dict, sparsity_percent, device, add_tie_break_noise, tie_break_noise_scale)

    if min_layer_keep_ratio > 0:
        print(
            "Mask pooling: global with per-layer keep floor "
            f"(min_layer_keep_ratio={min_layer_keep_ratio}; remaining budget is global top-k)"
        )
    else:
        print("Mask pooling: global (single ranking across all scored weights)")

    print(f"\n=== Creating Exact Global Masks (target sparsity: {sparsity_percent}%) ===")

    if not scores_dict:
        raise ValueError(
            "create_mask_from_scores_gpu_efficient received an empty scores_dict. "
            "Upstream scoring produced no valid weight-score tensors."
        )

    # Normalize tensors onto target device and collect valid entries.
    valid_scores = {}
    total_params = 0
    chunked_min_numel = _get_env_int(
        "RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL",
        DEFAULT_CHUNKED_SELECTOR_MIN_NUMEL,
    )
    for name, score in scores_dict.items():
        if score is None or score.numel() == 0:
            continue
        total_params += score.numel()

    use_chunked = total_params >= chunked_min_numel
    selector_device = "cpu" if use_chunked else device
    if use_chunked:
        print(
            f"Chunked selector enabled for {total_params:,} parameters "
            f"(threshold: {chunked_min_numel:,}); moving scores to CPU for selection."
        )

    for name, score in scores_dict.items():
        if score is None or score.numel() == 0:
            continue
        s = score.to(device=selector_device, dtype=torch.float32)
        torch.nan_to_num_(s, nan=0.0, posinf=0.0, neginf=0.0)
        valid_scores[name] = s

    if not valid_scores or total_params == 0:
        raise ValueError(
            "No non-empty score tensors were available for global ranking. "
            "Check upstream scoring/mapping logic."
        )

    if use_chunked:
        masks = _create_mask_global_chunked(
            valid_scores,
            sparsity_percent=sparsity_percent,
            add_tie_break_noise=add_tie_break_noise,
            tie_break_noise_scale=tie_break_noise_scale,
            min_layer_keep_ratio=min_layer_keep_ratio,
        )
    else:
        masks = _create_mask_global_flat(
            valid_scores,
            sparsity_percent=sparsity_percent,
            add_tie_break_noise=add_tie_break_noise,
            tie_break_noise_scale=tie_break_noise_scale,
            min_layer_keep_ratio=min_layer_keep_ratio,
        )

    for name, score in scores_dict.items():
        if name in masks or score is None:
            continue
        masks[name] = torch.zeros_like(score, dtype=torch.float32).cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return masks


def compute_jaccard_similarity(pred_masks, true_masks):
    """
    Computes Jaccard similarity, spatial overlap, and cosine similarity
    between predicted and reference masks. Uses GPU for faster computation.
    """
    print("\n=== Computing Similarity Metrics ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    per_layer_jaccard = {}
    total_intersection = 0
    total_union = 0
    total_pred_kept = 0
    total_true_kept = 0
    
    for name in pred_masks.keys():
        if name not in true_masks:
            continue
        
        pred = pred_masks[name].to(device).bool()
        true = true_masks[name].to(device).bool()
        
        intersection = (pred & true).sum().item()
        union = (pred | true).sum().item()
        
        jaccard = intersection / union if union > 0 else 0.0
        per_layer_jaccard[name] = jaccard
        
        total_intersection += intersection
        total_union += union
        total_pred_kept += pred.sum().item()
        total_true_kept += true.sum().item()
        
        del pred, true
    
    torch.cuda.empty_cache()
    
    aggregate_jaccard = total_intersection / total_union if total_union > 0 else 0.0
    
    # Calculate additional metrics
    overlap_pred = total_intersection / total_pred_kept if total_pred_kept > 0 else 0.0
    overlap_true = total_intersection / total_true_kept if total_true_kept > 0 else 0.0
    cosine_similarity = total_intersection / ((total_pred_kept * total_true_kept) ** 0.5) if (total_pred_kept * total_true_kept) > 0 else 0.0
    
    if len(per_layer_jaccard) > 0:
        mean_jaccard = sum(per_layer_jaccard.values()) / len(per_layer_jaccard)
        min_jaccard = min(per_layer_jaccard.values())
        max_jaccard = max(per_layer_jaccard.values())
        
        print(f"Aggregate Jaccard Similarity:     {aggregate_jaccard:.4f}")
        print(f"Overlap (Intersect / Pred_Size):  {overlap_pred:.4f} ({overlap_pred*100:.1f}%)")
        print(f"Overlap (Intersect / True_Size):  {overlap_true:.4f} ({overlap_true*100:.1f}%)")
        print(f"Global Cosine Similarity:         {cosine_similarity:.4f}")
        print(f"Mean per-layer Jaccard:           {mean_jaccard:.4f}")
        
        return {
            "aggregate_jaccard": aggregate_jaccard,
            "mean_jaccard": mean_jaccard,
            "min_jaccard": min_jaccard,
            "max_jaccard": max_jaccard,
            "per_layer": per_layer_jaccard,
            "overlap_fraction_predicted": overlap_pred,
            "overlap_fraction_reference": overlap_true,
            "cosine_similarity": cosine_similarity
        }
    else:
        print("No matching layers found for similarity computation.")
        return None

def save_masks(masks, output_file, metadata=None):
    """Saves masks with optional metadata."""
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_dict = {"masks": masks}
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, output_file)
    print(f"\nMasks saved to: {output_file}")
    
    total_params = sum(m.numel() for m in masks.values())
    kept_params = sum(m.sum().item() for m in masks.values())
    actual_sparsity = 100.0 - (kept_params / total_params * 100)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Kept parameters: {int(kept_params):,}")
    print(f"Final sparsity: {actual_sparsity:.2f}%")

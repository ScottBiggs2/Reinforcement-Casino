import math

import torch

from src.utils.mask_utils import create_mask_from_scores_gpu_efficient, pooling_metadata


def expected_keep_count(total_params: int, sparsity_percent: float) -> int:
    keep_percent = 100.0 - sparsity_percent
    return max(1, min(total_params, int(keep_percent / 100.0 * total_params)))


def main() -> None:
    sparsity_percent = 80.0
    min_layer_keep_ratio = 0.10
    scores = {
        "layer_a.weight": torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=torch.float32),
        "layer_b.weight": torch.tensor([[1.0, 0.5], [0.4, 0.3]], dtype=torch.float32),
        "layer_c.weight": torch.tensor([[0.6, 0.2]], dtype=torch.float32),
    }

    total_params = sum(score.numel() for score in scores.values())
    keep_count = expected_keep_count(total_params, sparsity_percent)
    expected_local_floors = {
        name: int(min_layer_keep_ratio * score.numel()) for name, score in scores.items()
    }
    requested_floor_total = sum(expected_local_floors.values())
    assert requested_floor_total < keep_count, "Smoke test requires an unscaled floor allocation"

    masks = create_mask_from_scores_gpu_efficient(
        scores,
        sparsity_percent,
        device="cpu",
        add_tie_break_noise=False,
        min_layer_keep_ratio=min_layer_keep_ratio,
        local_pool=False,
    )

    kept_total = sum(int(mask.sum().item()) for mask in masks.values())
    assert kept_total == keep_count, f"Expected {keep_count} kept params, found {kept_total}"

    for name, floor in expected_local_floors.items():
        kept = int(masks[name].sum().item())
        assert kept >= floor, f"{name} kept {kept}, below floor {floor}"

    metadata = pooling_metadata(
        local_pool=False,
        min_layer_keep_ratio=min_layer_keep_ratio,
    )
    assert metadata["pooling_mode"] == "global_with_layer_floor"
    assert math.isclose(metadata["min_layer_keep_ratio"], min_layer_keep_ratio)

    print("Hybrid selector smoke passed.")
    print(f"  total_params={total_params}")
    print(f"  keep_count={keep_count}")
    print(f"  min_layer_keep_ratio={min_layer_keep_ratio}")
    for name, mask in masks.items():
        print(f"  {name}: kept={int(mask.sum().item())}/{mask.numel()}")


if __name__ == "__main__":
    main()

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> torch.Tensor:
    """Train a one-layer logistic probe and return abs(weight) neuron scores."""
    x = features.float().to(device)
    y = labels.float().to(device)

    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)

    if y.unique().numel() < 2:
        return torch.zeros(x.shape[1], dtype=torch.float32)

    # Standardize features to stabilize probe fitting.
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    probe = nn.Linear(x.shape[1], 1, bias=True).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        logits = probe(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    weight = probe.weight.detach().squeeze(0).abs().float().cpu()
    return weight


def compute_cav_scores(
    labeled_activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Per-layer CAV scores from labeled activations."""
    scores: Dict[str, torch.Tensor] = {}
    for layer_name, (x, y) in labeled_activations.items():
        layer_scores = train_linear_probe(
            features=x,
            labels=y,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        scores[layer_name] = layer_scores
    return scores

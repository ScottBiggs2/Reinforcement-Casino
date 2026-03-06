import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class FeatureExtractor:
    """Capture pooled activations from linear layers during forward passes."""

    def __init__(self, model: nn.Module, mlp_only: bool = True):
        self.model = model
        self.mlp_only = mlp_only
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.module_to_weight: Dict[str, str] = {}
        self.activation_sums: Dict[str, torch.Tensor] = {}
        self.activation_counts: Dict[str, int] = defaultdict(int)
        self.labeled_acts: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.labeled_targets: Dict[str, List[torch.Tensor]] = defaultdict(list)

        self._collect_mode: Optional[str] = None
        self._current_label: Optional[int] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.mlp_only and "mlp" not in module_name.lower():
                continue

            weight_name = f"{module_name}.weight" if module_name else "weight"
            self.module_to_weight[module_name] = weight_name
            handle = module.register_forward_hook(self._build_hook(module_name))
            self.handles.append(handle)

    def _build_hook(self, module_name: str):
        def hook(_module, _inputs, output):
            if self._collect_mode is None:
                return

            if isinstance(output, tuple):
                output = output[0]
            if not torch.is_tensor(output):
                return

            pooled = _pool_hidden(output)
            if pooled.numel() == 0:
                return

            if self._collect_mode == "activation":
                layer_sum = pooled.abs().sum(dim=0).detach().cpu()
                if module_name not in self.activation_sums:
                    self.activation_sums[module_name] = layer_sum
                else:
                    self.activation_sums[module_name] += layer_sum
                self.activation_counts[module_name] += pooled.shape[0]
                return

            if self._collect_mode == "labeled" and self._current_label is not None:
                self.labeled_acts[module_name].append(pooled.detach().cpu())
                targets = torch.full((pooled.shape[0],), self._current_label, dtype=torch.long)
                self.labeled_targets[module_name].append(targets)

        return hook

    def collect_activation_stats_start(self) -> None:
        self._collect_mode = "activation"
        self._current_label = None

    def collect_labeled_start(self, label: int) -> None:
        self._collect_mode = "labeled"
        self._current_label = int(label)

    def collect_stop(self) -> None:
        self._collect_mode = None
        self._current_label = None

    def get_activation_scores(self) -> Dict[str, torch.Tensor]:
        scores: Dict[str, torch.Tensor] = {}
        for module_name, total in self.activation_sums.items():
            count = max(1, self.activation_counts[module_name])
            scores[module_name] = total / count
        return scores

    def get_labeled_activations(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for module_name in self.labeled_acts.keys():
            xs = self.labeled_acts[module_name]
            ys = self.labeled_targets[module_name]
            if not xs or not ys:
                continue
            out[module_name] = (torch.cat(xs, dim=0), torch.cat(ys, dim=0))
        return out

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Pool to [batch, hidden] for classifier-friendly features."""
    if hidden.dim() == 2:
        return hidden
    if hidden.dim() >= 3:
        return hidden.mean(dim=1)
    return hidden.reshape(hidden.shape[0], -1)

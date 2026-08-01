"""Score MLP neurons with linear probes over chosen vs rejected activations."""

from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class CAVProbeScorer:
    """Train one logistic probe per layer and convert scores into masks."""

    def _probe_one_layer(
        self,
        pos: np.ndarray,
        neg: np.ndarray,
        mag_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        X = np.concatenate([pos, neg], axis=0)
        y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=300,
            random_state=42,
        )
        clf.fit(X_sc, y)

        cav = np.abs(clf.coef_[0])
        mag = np.mean(np.abs(pos), axis=0)

        cav_norm = cav / (cav.max() + 1e-8)
        mag_norm = mag / (mag.max() + 1e-8)

        combined = cav_norm + mag_weight * mag_norm

        metrics: Dict[str, Any] = {
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "train_accuracy": float(clf.score(X_sc, y)),
            "coef_l1_norm": float(np.abs(clf.coef_).sum()),
            "cv_accuracy_mean": None,
        }
        if len(y) >= 12:
            try:
                n_splits = min(3, max(2, len(y) // 6))
                cv = cross_val_score(
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=1.0,
                        max_iter=300,
                        random_state=42,
                    ),
                    X_sc,
                    y,
                    cv=n_splits,
                )
                metrics["cv_accuracy_mean"] = float(np.mean(cv))
            except Exception:
                pass

        return torch.tensor(combined, dtype=torch.float32), metrics

    def score(self, positive_acts, negative_acts, mag_weight: float = 1.0):
        """Return per-neuron scores, optionally blended with activation magnitude."""
        scores, _ = self.score_with_probe_report(positive_acts, negative_acts, mag_weight)
        return scores

    def score_with_probe_report(
        self, positive_acts, negative_acts, mag_weight: float = 1.0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Return (neuron_scores dict, probe_report) for interpretation / JSON export."""
        neuron_scores = {}
        layer_names = sorted(set(positive_acts) & set(negative_acts))
        print(f"[CAVProbeScorer] Scoring {len(layer_names)} layers (mag_weight={mag_weight})...")

        layers_meta: Dict[str, Any] = {}
        for layer_name in layer_names:
            pos = positive_acts[layer_name].numpy()
            neg = negative_acts[layer_name].numpy()
            combined, meta = self._probe_one_layer(pos, neg, mag_weight)
            neuron_scores[layer_name] = combined
            layers_meta[layer_name] = meta

        print("[CAVProbeScorer] Probe training done.")
        report = {
            "version": 1,
            "mag_weight": float(mag_weight),
            "layers": layers_meta,
        }
        return neuron_scores, report

    def scores_to_masks(self, neuron_scores, model, sparsity_percent=90.0, local_pool=False):
        """Broadcast neuron scores to `gate_proj`, `up_proj`, and `down_proj` masks.

        Args:
            local_pool: If False (default), use a single global threshold across all
                layers — neurons compete globally, so high-signal layers keep more and
                low-signal layers keep fewer.  This avoids the flat per-layer sparsity
                that per-layer selection produces.
                If True, each layer independently keeps `keep_frac * intermediate_size`
                neurons (original behavior, uniform sparsity per layer).
        """
        keep_frac = 1.0 - sparsity_percent / 100.0

        # Collect the (name, param, hook_key, scores) tuples we'll need regardless of mode.
        candidates = []
        for name, param in model.named_parameters():
            if "mlp" not in name or len(param.shape) != 2 or ".weight" not in name:
                continue
            mlp_prefix = name.rsplit(".", 2)[0]
            hook_key   = mlp_prefix + ".down_proj"
            if hook_key not in neuron_scores:
                continue
            if "gate_proj" not in name and "up_proj" not in name and "down_proj" not in name:
                continue
            candidates.append((name, param, hook_key, neuron_scores[hook_key]))

        if not candidates:
            return {}

        if not local_pool:
            # --- Global selection: one threshold across all layers ---
            # Each hook_key (down_proj) represents one layer's neuron scores.
            # Deduplicate so each neuron is counted once in the global pool.
            seen_hooks = {}
            for name, param, hook_key, scores in candidates:
                if hook_key not in seen_hooks:
                    seen_hooks[hook_key] = scores

            flat_all = torch.cat([s for s in seen_hooks.values()])
            n_keep   = max(1, int(keep_frac * flat_all.numel()))
            threshold = torch.topk(flat_all, n_keep, largest=True, sorted=False).values.min().item()

            # Build per-hook keep_1d using the global threshold.
            keep_1d_per_hook = {
                hook_key: (scores >= threshold).to(dtype=torch.bool)
                for hook_key, scores in seen_hooks.items()
            }
        else:
            # --- Local selection: each layer independently ---
            keep_1d_per_hook = {}
            for name, param, hook_key, scores in candidates:
                if hook_key in keep_1d_per_hook:
                    continue
                intermediate_size = scores.shape[0]
                n_keep = max(1, int(keep_frac * intermediate_size))
                _, top_idx = torch.topk(scores, n_keep)
                keep_1d = torch.zeros(intermediate_size, dtype=torch.bool)
                keep_1d[top_idx] = True
                keep_1d_per_hook[hook_key] = keep_1d

        masks = {}
        for name, param, hook_key, scores in candidates:
            keep_1d = keep_1d_per_hook[hook_key]
            intermediate_size = scores.shape[0]

            if "gate_proj" in name or "up_proj" in name:
                assert param.shape[0] == intermediate_size, (
                    f"Shape mismatch: {name} row dim {param.shape[0]} "
                    f"!= intermediate_size {intermediate_size}"
                )
                mask = keep_1d.unsqueeze(1).expand(param.shape).clone()
            elif "down_proj" in name:
                assert param.shape[1] == intermediate_size, (
                    f"Shape mismatch: {name} col dim {param.shape[1]} "
                    f"!= intermediate_size {intermediate_size}"
                )
                mask = keep_1d.unsqueeze(0).expand(param.shape).clone()
            else:
                continue

            masks[name] = mask

        total  = sum(m.numel() for m in masks.values())
        kept   = sum(m.sum().item() for m in masks.values())
        actual = 100.0 * (1.0 - kept / total) if total > 0 else 0.0
        mode_str = "local (per-layer)" if local_pool else "global (cross-layer)"
        print(
            f"[CAVProbeScorer] {len(masks)} masks generated ({mode_str}). "
            f"Actual sparsity: {actual:.2f}%"
        )
        return masks


def compute_cav_scores(
    labeled_activations,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    normalize_per_layer: bool = True,
):
    """Compute per-neuron CAV scores from labeled activations.

    Parameters are kept for API compatibility with callers. `epochs`, `lr`,
    and `weight_decay` are currently unused because we fit sklearn logistic
    probes directly.

    Args:
        labeled_activations: dict[layer_name] -> (X, y)
            X: torch.Tensor [N, D], y: torch.Tensor [N] with binary labels.
        normalize_per_layer: If True, min-max normalize each layer's score to
            [0, 1] so deeper layers do not dominate global thresholding.

    Returns:
        dict[layer_name] -> torch.Tensor [D] of non-negative neuron scores.
    """
    del epochs, lr, weight_decay, device  # kept for signature compatibility

    neuron_scores = {}
    layer_names = sorted(labeled_activations.keys())
    print(f"[compute_cav_scores] Training probes for {len(layer_names)} layers...")

    for layer_name in layer_names:
        X, y = labeled_activations[layer_name]
        if X is None or y is None:
            continue

        X = X.detach().float().cpu()
        y = y.detach().long().cpu()

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] < 4:
            continue

        pos = int((y == 1).sum().item())
        neg = int((y == 0).sum().item())
        if pos == 0 or neg == 0:
            continue

        X_np = X.numpy()
        y_np = y.numpy()

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_np)

        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=300,
            random_state=42,
        )
        clf.fit(X_sc, y_np)

        cav = np.abs(clf.coef_[0]).astype(np.float32)

        if normalize_per_layer:
            cmin = float(cav.min())
            cmax = float(cav.max())
            if cmax > cmin:
                cav = (cav - cmin) / (cmax - cmin)
            else:
                cav = np.zeros_like(cav, dtype=np.float32)

        neuron_scores[layer_name] = torch.tensor(cav, dtype=torch.float32)

    print("[compute_cav_scores] Done.")
    return neuron_scores

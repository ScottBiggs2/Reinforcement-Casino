"""Score MLP neurons with linear probes over chosen vs rejected activations."""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CAVProbeScorer:
    """Train one logistic probe per layer and convert scores into masks."""

    def score(self, positive_acts, negative_acts, mag_weight: float = 1.0):
        """Return per-neuron scores, optionally blended with activation magnitude."""
        neuron_scores = {}
        layer_names = sorted(set(positive_acts) & set(negative_acts))
        print(f"[CAVProbeScorer] Scoring {len(layer_names)} layers (mag_weight={mag_weight})...")

        for layer_name in layer_names:
            pos = positive_acts[layer_name].numpy()   # [N_pos, D]
            neg = negative_acts[layer_name].numpy()   # [N_neg, D]

            X = np.concatenate([pos, neg], axis=0)
            y = np.array([1] * len(pos) + [0] * len(neg))

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

            cav = np.abs(clf.coef_[0])  # [D]

            mag = np.mean(np.abs(pos), axis=0)  # [D]

            cav_norm = cav / (cav.max() + 1e-8)
            mag_norm = mag / (mag.max() + 1e-8)

            combined = cav_norm + mag_weight * mag_norm
            neuron_scores[layer_name] = torch.tensor(combined, dtype=torch.float32)

        print("[CAVProbeScorer] Probe training done.")
        return neuron_scores

    def scores_to_masks(self, neuron_scores, model, sparsity_percent=90.0):
        """Broadcast neuron scores to `gate_proj`, `up_proj`, and `down_proj` masks."""
        keep_frac = 1.0 - sparsity_percent / 100.0
        masks = {}

        for name, param in model.named_parameters():
            if "mlp" not in name or len(param.shape) != 2 or ".weight" not in name:
                continue

            mlp_prefix = name.rsplit(".", 2)[0]   # "model.layers.X.mlp"
            hook_key   = mlp_prefix + ".down_proj"

            if hook_key not in neuron_scores:
                continue

            scores = neuron_scores[hook_key]           # [intermediate_size]
            intermediate_size = scores.shape[0]
            n_keep = max(1, int(keep_frac * intermediate_size))

            _, top_idx = torch.topk(scores, n_keep)
            keep_1d = torch.zeros(intermediate_size)
            keep_1d[top_idx] = 1.0

            if "gate_proj" in name or "up_proj" in name:
                # [intermediate_size, hidden_size] → keep row n
                assert param.shape[0] == intermediate_size, (
                    f"Shape mismatch: {name} row dim {param.shape[0]} "
                    f"!= intermediate_size {intermediate_size}"
                )
                mask = keep_1d.unsqueeze(1).expand(param.shape).clone()

            elif "down_proj" in name:
                # [hidden_size, intermediate_size] → keep col n
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
        print(
            f"[CAVProbeScorer] {len(masks)} masks generated. "
            f"Actual sparsity: {actual:.2f}%"
        )
        return masks

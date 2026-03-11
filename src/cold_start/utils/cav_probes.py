"""
CAV-based neuron scoring via linear probes (Kim et al., 2018).

For each MLP layer:
  - Positive class: activations from *chosen* (preferred) responses
  - Negative class: activations from *rejected* responses
  - Train a Logistic Regression probe; neuron importance = |coef_|

Combined scoring (subnetwork-aware):
  Pure CAV only captures *discriminative* neurons (chosen ≠ rejected), but
  misses *shared* neurons that are required for the task yet fire on both
  chosen and rejected (e.g. general reasoning circuits used by all tasks).

  To avoid dropping shared weights we combine two signals:
    cav_score  = |probe coef|            — discriminative signal
    mag_score  = mean |act| on chosen    — task-presence signal

  Both are normalised to [0, 1] per layer, then summed:
    final = norm(cav_score) + mag_weight * norm(mag_score)

  mag_weight=0 → pure CAV (original); mag_weight=1 → equal blend (default).

Weight mapping for SwiGLU-style MLP (gate_proj, up_proj, down_proj):
  gate_proj.weight  [intermediate, hidden]:  row n = neuron n
  up_proj.weight    [intermediate, hidden]:  row n = neuron n
  down_proj.weight  [hidden, intermediate]:  col n = neuron n
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CAVProbeScorer:
    """
    Trains one lightweight logistic probe per MLP layer and converts
    probe coefficients into 2-D binary weight masks.
    """

    # ------------------------------------------------------------------
    def score(self, positive_acts, negative_acts, mag_weight: float = 1.0):
        """
        Args:
            positive_acts: dict {layer_name: Tensor[N_pos, intermediate_size]}
            negative_acts: dict {layer_name: Tensor[N_neg, intermediate_size]}
            mag_weight:    how much to weight the task-presence signal vs the
                           discriminative CAV signal (default 1.0 = equal blend).
                           Set to 0.0 to reproduce the original pure-CAV behaviour.

        Returns:
            neuron_scores: dict {layer_name: Tensor[intermediate_size]}
                           (higher = more task-relevant)
        """
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

            # --- Discriminative signal: |CAV coef| ----------------------------
            # High for neurons that separate chosen from rejected.
            # Misses neurons shared by both classes but still needed for the task.
            cav = np.abs(clf.coef_[0])  # [D]

            # --- Task-presence signal: mean |activation| on chosen ------------
            # High for ALL neurons that fire during the target task, including
            # neurons shared with the rejected class (e.g. general reasoning
            # circuits). This is the key fix for the shared-subnetwork pitfall.
            mag = np.mean(np.abs(pos), axis=0)  # [D]

            # Normalise both to [0, 1] per layer so neither dominates by scale
            cav_norm = cav / (cav.max() + 1e-8)
            mag_norm = mag / (mag.max() + 1e-8)

            combined = cav_norm + mag_weight * mag_norm
            neuron_scores[layer_name] = torch.tensor(combined, dtype=torch.float32)

        print("[CAVProbeScorer] Probe training done.")
        return neuron_scores

    # ------------------------------------------------------------------
    def scores_to_masks(self, neuron_scores, model, sparsity_percent=90.0):
        """
        Broadcast per-neuron CAV importance scores to 2-D weight masks.

        Args:
            neuron_scores: dict {hook_key ending in 'down_proj': Tensor[intermediate]}
            model:          the transformer model (used to iterate named_parameters)
            sparsity_percent: % of weights set to 0

        Returns:
            masks: dict {param_name: float Tensor, same shape as param}  (1=keep, 0=prune)
        """
        keep_frac = 1.0 - sparsity_percent / 100.0
        masks = {}

        for name, param in model.named_parameters():
            if "mlp" not in name or len(param.shape) != 2 or ".weight" not in name:
                continue

            # Derive the hook key: "model.layers.X.mlp.down_proj"
            # param name: "model.layers.X.mlp.{gate|up|down}_proj.weight"
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

"""
Hook-based feature extraction for cold start inference.

Captures MLP intermediate activations (the input to down_proj, i.e., the
output of silu(gate_proj(x)) * up_proj(x)) during forward passes with
NO weight updates.
"""

import torch
import torch.nn as nn
from collections import defaultdict


class FeatureExtractor:
    """
    Registers forward hooks on every down_proj Linear layer in the model.

    The input to down_proj is the neuron activation vector for that MLP block:
        intermediate = silu(gate_proj(x)) * up_proj(x)  [batch, seq, intermediate]
    We record its mean over (batch, seq) → one vector per forward pass.

    Usage:
        extractor = FeatureExtractor()
        extractor.register(model)
        acts = extractor.collect(model, tokenizer, texts, device)
        extractor.remove()
    """

    def __init__(self):
        self.activations = defaultdict(list)   # {layer_name: [tensor per batch]}
        self._hooks = []

    # ------------------------------------------------------------------
    def register(self, model):
        """Attach hooks to every down_proj layer. Returns self for chaining."""
        for name, module in model.named_modules():
            if name.endswith("down_proj") and isinstance(module, nn.Linear):
                def _make_hook(lname):
                    def _hook(mod, inp, out):
                        # inp[0]: [batch, seq_len, intermediate_size]
                        act = inp[0].detach().float()
                        self.activations[lname].append(act.mean(dim=(0, 1)).cpu())
                    return _hook
                h = module.register_forward_hook(_make_hook(name))
                self._hooks.append(h)

        print(f"[FeatureExtractor] Hooked {len(self._hooks)} down_proj layers.")
        return self

    # ------------------------------------------------------------------
    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print("[FeatureExtractor] Hooks removed.")

    def clear(self):
        self.activations.clear()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect(self, model, tokenizer, texts, device, max_length=512, batch_size=8):
        """
        Run inference on `texts`, returning stacked intermediate activations.

        Returns:
            dict {layer_name: Tensor[N_batches, intermediate_size]}
        """
        self.clear()
        model.eval()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

        # Stack all per-batch vectors → [N, intermediate_size]
        return {
            name: torch.stack(acts_list, dim=0)
            for name, acts_list in self.activations.items()
        }

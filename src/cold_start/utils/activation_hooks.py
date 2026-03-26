"""Collect pooled MLP activations with forward hooks."""

import torch
import torch.nn as nn
from collections import defaultdict


def infer_model_input_device(model) -> torch.device:
    """Infer the device that should receive model inputs."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str) and first_device not in {"cpu", "disk"}:
            return torch.device(first_device)

    return next(model.parameters()).device


class FeatureExtractor:
    """Capture one pooled activation vector per sample for each `down_proj` layer."""

    def __init__(self):
        self.activations = defaultdict(list)
        self._hooks = []
        self._current_attention_mask = None

    def register(self, model):
        """Attach hooks to every down_proj layer. Returns self for chaining."""
        for name, module in model.named_modules():
            if name.endswith("down_proj") and isinstance(module, nn.Linear):
                def _make_hook(lname):
                    def _hook(mod, inp, out):
                        act = inp[0].detach().float()

                        # Global mean pool over real (non-padding) tokens.
                        if self._current_attention_mask is not None:
                            mask = self._current_attention_mask
                            if mask.dim() == 2 and mask.shape[:2] == act.shape[:2]:
                                mask = mask.to(act.device).unsqueeze(-1).float()
                                denom = mask.sum(dim=1).clamp_min(1.0)
                                pooled = (act * mask).sum(dim=1) / denom
                            else:
                                pooled = act.mean(dim=1)
                        else:
                            pooled = act.mean(dim=1)

                        self.activations[lname].append(pooled.cpu())
                    return _hook
                h = module.register_forward_hook(_make_hook(name))
                self._hooks.append(h)

        print(f"[FeatureExtractor] Hooked {len(self._hooks)} down_proj layers.")
        return self

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print("[FeatureExtractor] Hooks removed.")

    def clear(self):
        self.activations.clear()
        self._current_attention_mask = None

    @torch.no_grad()
    def collect(self, model, tokenizer, texts, device, max_length=512, batch_size=8):
        """Run inference and return `{layer_name: Tensor[n_samples, hidden]}`."""
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
            self._current_attention_mask = attention_mask
            model(input_ids=input_ids, attention_mask=attention_mask)
            self._current_attention_mask = None

        return {
            name: torch.cat(acts_list, dim=0)
            for name, acts_list in self.activations.items()
        }

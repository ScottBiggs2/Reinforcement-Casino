"""Score MLP weights with one backward pass of gradient saliency."""

import torch
import torch.nn.functional as F


class SNIPScorer:
    """Compute `|grad * weight|` for each MLP weight matrix."""

    def score(self, model, tokenizer, chosen_texts, device, max_length=512, batch_size=8):
        """Return per-parameter saliency scores without updating weights."""
        model.eval()
        model.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=device)
        n_batches  = 0

        for i in range(0, len(chosen_texts), batch_size):
            batch = chosen_texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss = total_loss + out.loss
            n_batches  += 1

        (total_loss / max(n_batches, 1)).backward()

        scores = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "mlp" not in name or len(param.shape) != 2:
                    continue
                if param.grad is None:
                    continue
                scores[name] = (param.grad.abs() * param.detach().abs()).float().cpu()

        model.zero_grad(set_to_none=True)
        model.eval()

        print(f"[SNIPScorer] Scored {len(scores)} MLP weight matrices.")
        return scores

    def scores_to_masks(self, scores, sparsity_percent=90.0):
        """Keep the global top-k saliency weights and drop the rest."""
        keep_frac = 1.0 - sparsity_percent / 100.0

        all_scores = torch.cat([s.flatten() for s in scores.values()])
        n_keep     = max(1, int(keep_frac * all_scores.numel()))
        threshold, _ = torch.topk(all_scores, n_keep)
        threshold  = threshold.min().item()

        masks = {name: (score >= threshold).float() for name, score in scores.items()}

        total  = sum(m.numel() for m in masks.values())
        kept   = sum(m.sum().item() for m in masks.values())
        actual = 100.0 * (1.0 - kept / total) if total > 0 else 0.0
        print(
            f"[SNIPScorer] {len(masks)} masks, threshold={threshold:.2e}, "
            f"actual sparsity={actual:.2f}%"
        )
        return masks

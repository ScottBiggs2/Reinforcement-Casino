"""Score MLP weights with one backward pass of gradient saliency."""

import torch
import torch.nn.functional as F

from src.utils.mask_utils import create_mask_from_scores_gpu_efficient


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

    def scores_to_masks(self, scores, sparsity_percent=90.0, local_pool=False):
        """Keep top-k saliency weights.

        Args:
            local_pool: If False (default), one global threshold across all layers.
                        If True, each weight matrix independently keeps keep_frac elements.
        """
        masks = create_mask_from_scores_gpu_efficient(
            scores,
            sparsity_percent=sparsity_percent,
            device="cpu",
            local_pool=local_pool,
            min_layer_keep_ratio=0.0,
        )

        total  = sum(m.numel() for m in masks.values())
        kept   = sum(m.sum().item() for m in masks.values())
        actual = 100.0 * (1.0 - kept / total) if total > 0 else 0.0
        mode   = "local (per-layer)" if local_pool else "global (cross-layer)"
        print(f"[SNIPScorer] {len(masks)} masks ({mode}), actual sparsity={actual:.2f}%")
        return masks


def _sequence_logprob(logits, input_ids, attention_mask):
    """Return per-sample summed log-probability for non-padding next tokens."""
    # Standard causal LM shift.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * shift_mask
    return token_logp.sum(dim=-1)


def dpo_style_preference_loss(
    chosen_logits,
    chosen_ids,
    chosen_mask,
    rejected_logits,
    rejected_ids,
    rejected_mask,
    beta: float = 1.0,
):
    """A simple DPO-style loss using model-only chosen-vs-rejected margins."""
    chosen_lp = _sequence_logprob(chosen_logits, chosen_ids, chosen_mask)
    rejected_lp = _sequence_logprob(rejected_logits, rejected_ids, rejected_mask)
    margin = chosen_lp - rejected_lp
    return -F.logsigmoid(beta * margin).mean()


def compute_snip_scores(
    model,
    dataloader,
    device: str,
    num_batches: int,
    mlp_only: bool = False,
):
    """Compute SNIP scores for model weights from DPO-style preference gradients."""
    model.train()
    model.zero_grad(set_to_none=True)

    seen = 0
    for idx, batch in enumerate(dataloader):
        if idx >= num_batches:
            break

        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
        rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

        loss = dpo_style_preference_loss(
            chosen_logits=chosen_logits,
            chosen_ids=chosen_ids,
            chosen_mask=chosen_mask,
            rejected_logits=rejected_logits,
            rejected_ids=rejected_ids,
            rejected_mask=rejected_mask,
        )
        loss.backward()
        seen += 1

    scores = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if mlp_only and "mlp" not in name:
                continue
            if param.dim() != 2:
                continue
            if param.grad is None:
                continue
            scores[name] = (param.grad.abs() * param.detach().abs()).float().cpu()

    model.zero_grad(set_to_none=True)
    model.eval()
    print(f"[compute_snip_scores] Scored {len(scores)} matrices over {seen} batches.")
    return scores

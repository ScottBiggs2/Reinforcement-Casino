from typing import Dict

import torch
import torch.nn.functional as F


def sequence_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Token NLL averaged per sequence."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    seq_loss = (token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)
    return seq_loss


def dpo_style_preference_loss(
    chosen_logits: torch.Tensor,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_logits: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> torch.Tensor:
    """A lightweight DPO-style ranking objective for scoring saliency."""
    chosen_nll = sequence_nll(chosen_logits, chosen_ids, chosen_mask)
    rejected_nll = sequence_nll(rejected_logits, rejected_ids, rejected_mask)
    margin = rejected_nll - chosen_nll
    return -F.logsigmoid(margin).mean()


def compute_snip_scores(
    model,
    dataloader,
    device: str,
    num_batches: int,
    mlp_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute SNIP score |w * grad| from a small preference batch set."""
    model.train()
    model.zero_grad(set_to_none=True)

    batch_count = 0
    for batch in dataloader:
        if batch_count >= num_batches:
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
        batch_count += 1

    scores: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if param.dim() != 2 or not name.endswith(".weight"):
                continue
            if mlp_only and "mlp" not in name.lower():
                continue
            scores[name] = (param.detach().float().abs() * param.grad.detach().float().abs()).cpu()

    model.zero_grad(set_to_none=True)
    model.eval()
    return scores

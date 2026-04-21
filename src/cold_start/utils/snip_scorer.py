"""Score MLP weights from gradient saliency (SNIP): mean-loss gradients via per-step backward."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    pooling_metadata,
)

# CLI / metadata string values (match inference_mask_finder --snip-objective)
SNIP_OBJECTIVE_LM = "lm"
SNIP_OBJECTIVE_DPO_PREFERENCE = "dpo_preference"


def build_snip_masks_from_scores(
    scores: Dict[str, torch.Tensor],
    *,
    sparsity_percent: float,
    device: str = "cpu",
    local_pool: bool = False,
    min_layer_keep_ratio: float = DEFAULT_MIN_LAYER_KEEP_RATIO,
    add_tie_break_noise: bool = True,
) -> Dict[str, torch.Tensor]:
    """Top-k binary masks from SNIP saliency scores (``torch.bool``); same pooling rules as other cold masks."""
    out = create_mask_from_scores_gpu_efficient(
        scores,
        sparsity_percent=sparsity_percent,
        device=device,
        add_tie_break_noise=add_tie_break_noise,
        tie_break_noise_scale=1e-6,
        min_layer_keep_ratio=min_layer_keep_ratio,
        local_pool=local_pool,
    )
    assert all(v.dtype == torch.bool for v in out.values()), "SNIP masks must be torch.bool"
    return out


def snip_save_metadata(
    *,
    snip_objective: str,
    sparsity_percent: float,
    local_pool: bool,
    min_layer_keep_ratio: float,
    preference_beta: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Metadata fields for torch.save alongside masks (reproducibility)."""
    meta = {
        "method": "snip",
        "snip_objective": snip_objective,
        "sparsity_percent": float(sparsity_percent),
        **pooling_metadata(
            local_pool=local_pool,
            min_layer_keep_ratio=min_layer_keep_ratio,
        ),
    }
    if preference_beta is not None:
        meta["preference_beta"] = float(preference_beta)
    if extra:
        meta.update(extra)
    return meta


class SNIPScorer:
    """Compute `|grad * weight|` for each MLP weight matrix."""

    def score(
        self,
        model,
        tokenizer,
        chosen_texts,
        device,
        max_length=512,
        batch_size=8,
        mlp_only=False,
        *,
        gradient_checkpointing: bool = True,
        use_autocast: bool = True,
    ):
        """Return per-parameter SNIP saliency (|grad * w|) for the mean CE loss over batches.

        Uses **per-batch backward** with scale ``1/K`` so gradients match mean loss **without**
        fusing all forwards into one autograd graph (which would scale VRAM with batch count).
        """
        dev = torch.device(device)
        use_cuda = dev.type == "cuda" and torch.cuda.is_available()

        n_batches = (len(chosen_texts) + batch_size - 1) // batch_size if chosen_texts else 0
        if n_batches == 0:
            print("[SNIPScorer] No calibration texts; returning empty scores.")
            return {}

        gc_was_on = bool(getattr(model, "is_gradient_checkpointing", False))
        we_turned_gc_on = False
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable") and not gc_was_on:
            model.gradient_checkpointing_enable()
            we_turned_gc_on = True

        model.train()
        model.zero_grad(set_to_none=True)

        amp_dtype = torch.bfloat16
        if use_cuda:
            try:
                p0 = next(model.parameters())
                if p0.dtype == torch.float16:
                    amp_dtype = torch.float16
            except StopIteration:
                pass

        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_autocast))
            if use_cuda and use_autocast
            else nullcontext()
        )

        k = float(n_batches)
        seen = 0
        try:
            for i in range(0, len(chosen_texts), batch_size):
                batch = chosen_texts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = enc["input_ids"].to(device, non_blocking=True)
                attention_mask = enc["attention_mask"].to(device, non_blocking=True)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                with ctx:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss_scaled = out.loss / k

                loss_scaled.backward()

                del out, loss_scaled, input_ids, attention_mask, labels, enc, batch
                if use_cuda:
                    torch.cuda.empty_cache()

                seen += 1
        finally:
            if we_turned_gc_on and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

        scores = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if mlp_only and "mlp" not in name:
                    continue
                if len(param.shape) != 2:
                    continue
                if param.grad is None:
                    continue
                scores[name] = (param.grad.abs() * param.detach().abs()).float().cpu()

        model.zero_grad(set_to_none=True)
        model.eval()

        print(
            f"[SNIPScorer] Scored {len(scores)} weight matrices over {seen} batches "
            f"(K={n_batches}, grad_ckpt={gradient_checkpointing}, autocast={use_autocast and use_cuda})."
        )
        return scores

    def scores_to_masks(
        self,
        scores,
        sparsity_percent=90.0,
        local_pool=False,
        min_layer_keep_ratio: float = DEFAULT_MIN_LAYER_KEEP_RATIO,
    ):
        """Keep top-k saliency weights (delegates to shared pooling logic)."""
        masks = build_snip_masks_from_scores(
            scores,
            sparsity_percent=sparsity_percent,
            device="cpu",
            local_pool=local_pool,
            min_layer_keep_ratio=min_layer_keep_ratio,
            add_tie_break_noise=True,
        )

        total = sum(m.numel() for m in masks.values())
        kept = sum(m.sum().item() for m in masks.values())
        actual = 100.0 * (1.0 - kept / total) if total > 0 else 0.0
        mode = "local (per-layer)" if local_pool else "global (cross-layer)"
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


def _collect_preference_batch_sizes(dataloader, num_batches: int) -> List[int]:
    """First pass: batch sizes only (for global-mean gradient scaling)."""
    sizes: List[int] = []
    for idx, batch in enumerate(dataloader):
        if idx >= num_batches:
            break
        sizes.append(int(batch["chosen_input_ids"].shape[0]))
    return sizes


def compute_snip_scores(
    model,
    dataloader,
    device: str,
    num_batches: int,
    mlp_only: bool = False,
    preference_beta: float = 1.0,
    *,
    gradient_checkpointing: bool = True,
    use_autocast: bool = True,
):
    """Compute SNIP scores from gradients of ``dpo_style_preference_loss`` (pairwise preference).

    Uses gradient checkpointing (optional) and bf16 autocast on CUDA to limit activation memory.
    Micro-batches should use small ``batch_size`` in the DataLoader (default 1 in callers) so each
    step only builds one chosen+rejected graph at moderate width.

    Gradients are accumulated with weights ``batch_size / N_total`` so ``.grad`` matches the gradient
    of the **global mean** loss over all processed pairs (same as one backward on mean loss).
    """
    dev = torch.device(device)
    use_cuda = dev.type == "cuda" and torch.cuda.is_available()

    batch_sizes = _collect_preference_batch_sizes(dataloader, num_batches)
    if not batch_sizes:
        print("[compute_snip_scores] Warning: no batches; returning empty scores.")
        return {}

    n_total = sum(batch_sizes)
    gc_was_on = bool(getattr(model, "is_gradient_checkpointing", False))
    we_turned_gc_on = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable") and not gc_was_on:
        model.gradient_checkpointing_enable()
        we_turned_gc_on = True

    model.train()
    model.zero_grad(set_to_none=True)

    amp_dtype = torch.bfloat16
    if use_cuda:
        try:
            p0 = next(model.parameters())
            if p0.dtype == torch.float16:
                amp_dtype = torch.float16
        except StopIteration:
            pass

    seen = 0
    try:
        step_idx = 0
        for batch in dataloader:
            if step_idx >= num_batches:
                break
            bs = batch_sizes[step_idx]
            scale = float(bs) / float(n_total)

            chosen_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
            chosen_mask = batch["chosen_attention_mask"].to(device, non_blocking=True)
            rejected_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
            rejected_mask = batch["rejected_attention_mask"].to(device, non_blocking=True)

            ctx = (
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_autocast))
                if use_cuda and use_autocast
                else nullcontext()
            )

            with ctx:
                chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
                rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

                loss = dpo_style_preference_loss(
                    chosen_logits=chosen_logits,
                    chosen_ids=chosen_ids,
                    chosen_mask=chosen_mask,
                    rejected_logits=rejected_logits,
                    rejected_ids=rejected_ids,
                    rejected_mask=rejected_mask,
                    beta=preference_beta,
                )
                loss_weighted = loss * scale

            loss_weighted.backward()

            del chosen_logits, rejected_logits, loss, loss_weighted
            del chosen_ids, chosen_mask, rejected_ids, rejected_mask
            if use_cuda:
                torch.cuda.empty_cache()

            seen += 1
            step_idx += 1

    finally:
        if we_turned_gc_on and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

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
    print(
        f"[compute_snip_scores] Scored {len(scores)} matrices over {seen} batches "
        f"(N_total={n_total}, grad_ckpt={gradient_checkpointing}, autocast={use_autocast and use_cuda})."
    )
    return scores

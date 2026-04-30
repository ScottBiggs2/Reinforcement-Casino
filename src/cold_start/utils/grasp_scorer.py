"""Score weights from gradient signal preservation (GRaSP)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    pooling_metadata,
)
from src.cold_start.utils.snr_weighting import (
    GradientSNRAccumulator,
    SNRConfig,
    summarize_multiplier_stats,
)

# Reuse the same objectives as SNIP
GRASP_OBJECTIVE_LM = "lm"
GRASP_OBJECTIVE_DPO_PREFERENCE = "dpo_preference"


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


class GRaSPScorer:
    """
    Compute `- (H * g) * w` for each weight matrix.
    
    Implements a two-pass memory-efficient approach to compute the Hessian-gradient product (HGP).
    """

    def score(
        self,
        model,
        tokenizer,
        calibration_texts, # For DPO: List of pairs. For LM: List of strings.
        device,
        max_length=512,
        batch_size=1,
        mlp_only=False,
        *,
        objective: str = GRASP_OBJECTIVE_LM,
        gradient_checkpointing: bool = True,
        use_autocast: bool = True,
        response_masks: Optional[torch.Tensor] = None,
        preference_beta: float = 1.0,
        dataloader = None, # If provided, uses the dataloader directly (DPO/GRPO modes)
        score_snr: Literal["off", "per_tensor", "per_weight"] = "off",
        score_snr_eps: float = 1e-8,
        score_snr_transform: Literal["identity", "log1p", "clamp"] = "log1p",
        score_snr_clamp_min: float = 0.0,
        score_snr_clamp_max: float = 50.0,
        score_snr_ram_budget_gb: float = 8.0,
        score_snr_allow_large_ram: bool = False,
    ):
        dev = torch.device(device)
        use_cuda = dev.type == "cuda" and torch.cuda.is_available()

        # Preparation
        gc_was_on = bool(getattr(model, "is_gradient_checkpointing", False))
        we_turned_gc_on = False
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable") and not gc_was_on:
            # torch.utils.checkpoint reentrant mode is incompatible with torch.autograd.grad().
            # HF Transformers supports non-reentrant via gradient_checkpointing_kwargs in newer versions.
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                we_turned_gc_on = True
            except TypeError:
                # Older Transformers: cannot request non-reentrant. Enabling gradient checkpointing
                # would likely default to reentrant=True and crash when we call autograd.grad().
                # Prefer correctness over memory here; the caller can still reduce VRAM via batch_size=1.
                print(
                    "[GRaSP] WARNING: gradient_checkpointing_kwargs not supported by this Transformers version; "
                    "disabling gradient checkpointing to avoid reentrant checkpoint + autograd.grad() crash."
                )
                gradient_checkpointing = False

        model.train()
        
        amp_dtype = torch.bfloat16
        if use_cuda:
            try:
                p0 = next(model.parameters())
                if p0.dtype == torch.float16:
                    amp_dtype = torch.float16
            except StopIteration:
                pass

        # ---------------------------------------------------------------------
        # Pass 1: Compute the average gradient g_avg
        # ---------------------------------------------------------------------
        print(f"[GRaSP] Pass 1: Computing average gradient (objective={objective})...")
        model.zero_grad(set_to_none=True)

        named_params: List[tuple[str, torch.Tensor]] = []
        for name, param in model.named_parameters():
            if mlp_only and "mlp" not in name:
                continue
            if param.dim() != 2:
                continue
            if not param.requires_grad:
                continue
            named_params.append((name, param))
        params_to_score = [p for _, p in named_params]

        g_sum = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params_to_score]

        snr_accum = None
        if score_snr != "off":
            cfg = SNRConfig(
                mode=score_snr,
                eps=float(score_snr_eps),
                transform=score_snr_transform,
                clamp_min=float(score_snr_clamp_min),
                clamp_max=float(score_snr_clamp_max),
                ram_budget_gb=float(score_snr_ram_budget_gb),
                allow_large_ram=bool(score_snr_allow_large_ram),
            )
            snr_accum = GradientSNRAccumulator(cfg=cfg, params_by_name={n: p for n, p in named_params})
        
        # We need a way to iterate through the data
        if dataloader is not None:
            data_iter = dataloader
            num_samples = len(dataloader.dataset)
        else:
            # Fallback for simple list of texts
            num_samples = len(calibration_texts)
            data_iter = range(0, num_samples, batch_size)

        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_autocast))
            if use_cuda and use_autocast
            else nullcontext()
        )

        seen_samples = 0
        
        # Helper to get loss
        def get_loss(batch_data, i_batch):
            if objective == GRASP_OBJECTIVE_LM:
                if dataloader is not None:
                    # In DPO mode, chosen_texts are the calibration_texts
                    # But if objective is LM, we might be passed a DPO batch
                    input_ids = batch_data["chosen_input_ids"].to(device)
                    attention_mask = batch_data["chosen_attention_mask"].to(device)
                else:
                    batch = calibration_texts[batch_data : batch_data + batch_size]
                    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                if response_masks is not None and dataloader is None:
                    batch_rmask = response_masks[batch_data : batch_data + batch_size].to(device)
                    labels[batch_rmask == 0] = -100
                
                with ctx:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    return out.loss
            else:
                # DPO Preference
                chosen_ids = batch_data["chosen_input_ids"].to(device)
                chosen_mask = batch_data["chosen_attention_mask"].to(device)
                rejected_ids = batch_data["rejected_input_ids"].to(device)
                rejected_mask = batch_data["rejected_attention_mask"].to(device)
                with ctx:
                    chosen_logits = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
                    rejected_logits = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits
                    return dpo_style_preference_loss(
                        chosen_logits, chosen_ids, chosen_mask,
                        rejected_logits, rejected_ids, rejected_mask,
                        beta=preference_beta
                    )

        # Pass 1 Loop
        for i, batch_data in enumerate(data_iter):
            loss = get_loss(batch_data, i)
            grads = torch.autograd.grad(loss, params_to_score, retain_graph=False, create_graph=False)

            if snr_accum is not None:
                snr_accum.update_from_batch_grads({name: g for (name, _), g in zip(named_params, grads)})

            bs = batch_data["chosen_input_ids"].shape[0] if isinstance(batch_data, dict) else batch_size
            seen_samples += bs

            with torch.no_grad():
                for j, g in enumerate(grads):
                    if g is None:
                        continue
                    # loss is a mean over the batch -> multiply by bs to sum per-example gradients.
                    g_sum[j].add_(g.detach().cpu().float(), alpha=float(bs))

            if i % 10 == 0:
                print(f"  [Pass 1] Batch {i} processed...")

            del loss, grads

        # Normalize and store g_avg on CPU
        g_avg = [gs / float(max(seen_samples, 1)) for gs in g_sum]
        
        # ---------------------------------------------------------------------
        # Pass 2: Compute H * g_avg using HVPs
        # ---------------------------------------------------------------------
        print(f"[GRaSP] Pass 2: Computing Hessian-gradient product (seen_samples={seen_samples})...")
        hgp_avg = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params_to_score]
        
        # Move g_avg to device for HVP calculation
        g_avg_dev = [g.to(device) for g in g_avg]

        # Reset iterator
        if dataloader is not None:
            data_iter = dataloader
        else:
            data_iter = range(0, num_samples, batch_size)

        for i, batch_data in enumerate(data_iter):
            model.zero_grad(set_to_none=True)
            loss = get_loss(batch_data, i)
            
            # Compute current batch gradient g_B with create_graph=True
            grads_B = torch.autograd.grad(loss, params_to_score, create_graph=True)
            
            # Compute scalar product Z = sum(g_B * g_avg)
            # We must keep the graph for grads_B
            z = 0
            for g_B, g_A in zip(grads_B, g_avg_dev):
                z = z + (g_B * g_A).sum()
            
            # Compute grad(z) w.r.t params -> this is H_B * g_avg
            # We don't need create_graph=True here unless we wanted 3rd order
            hvp_B = torch.autograd.grad(z, params_to_score)
            
            bs = batch_data["chosen_input_ids"].shape[0] if isinstance(batch_data, dict) else batch_size
            
            # Accumulate
            for j, hvp in enumerate(hvp_B):
                hgp_avg[j] += hvp.detach().cpu().float() * (float(bs) / float(seen_samples))
            
            if i % 10 == 0:
                print(f"  [Pass 2] Batch {i} processed...")
            
            # Cleanup to save memory
            del loss, grads_B, z, hvp_B
            if use_cuda:
                torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Final: Compute scores S = - (H * g) * w
        # ---------------------------------------------------------------------
        print("[GRaSP] Computing final scores...")
        scores = {}
        param_names = [name for name, _ in named_params]

        multipliers = {}
        if snr_accum is not None:
            multipliers, summary = snr_accum.snr_multipliers()
            print(
                f"[GRaSP] SNR weighting enabled: mode={score_snr} "
                f"transform={score_snr_transform} eps={score_snr_eps} "
                f"summary={summarize_multiplier_stats(summary)}"
            )

        for name, p_cpu, hgp in zip(param_names, [p.detach().cpu().float() for p in params_to_score], hgp_avg):
            # S = - (H * g) * w
            s = -(hgp * p_cpu)
            if multipliers:
                m = multipliers.get(name)
                if m is not None:
                    s = s * (m if m.numel() > 1 else float(m.item()))
            scores[name] = s

        if we_turned_gc_on and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        model.zero_grad(set_to_none=True)
        model.eval()
        
        print(f"[GRaSP] Scored {len(scores)} matrices.")
        return scores


def grasp_save_metadata(
    *,
    grasp_objective: str,
    sparsity_percent: float,
    local_pool: bool,
    min_layer_keep_ratio: float,
    preference_beta: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Metadata fields for torch.save alongside masks (reproducibility)."""
    meta = {
        "method": "grasp",
        "grasp_objective": grasp_objective,
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

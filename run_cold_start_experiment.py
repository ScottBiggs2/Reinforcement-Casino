#!/usr/bin/env python3
"""
Cold-Start Experiment
=====================
End-to-end script:
  1. Generate cold-start mask (CAV method, no training)
  2. Train sparse DPO for 10 steps using cold-start mask
  3. Train sparse DPO for 10 steps using a warm-start mask (comparison)
  4. Train dense DPO for 10 steps (baseline)
  5. Print loss comparison table

Run:
  python run_cold_start_experiment.py
"""

import os
import sys
import json
import time
import torch
import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOTrainer, DPOConfig

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.cold_start.utils.activation_hooks import FeatureExtractor
from src.cold_start.utils.cav_probes import CAVProbeScorer
from src.cold_start.utils.snip_scorer import SNIPScorer
from src.utils.mask_manager import SparseMaskManager

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME     = "google/gemma-3-270m-it"
DATASET_NAME   = "qihoo360/Light-R1-DPOData"
N_TRAIN_STEPS  = 10
N_CAL_SAMPLES  = 64       # samples used for cold-start scoring
SUBSET_SIZE    = 100      # DPO training subset
SPARSITY       = 90.0     # % of MLP weights to prune
BATCH_SIZE     = 2
LR             = 5e-5
MAX_LENGTH     = 512
WARM_MASK_PATH = "masks/sparsity_97.5pct_momentum_w5_step40.pt"   # existing mask

COLD_MASK_CAV  = "masks/cold_start_cav_90pct.pt"
COLD_MASK_SNIP = "masks/cold_start_snip_90pct.pt"
RESULTS_DIR    = "results/cold_start_experiment"


# ============================================================
# Dataset helpers
# ============================================================

def _msg_to_text(x):
    if isinstance(x, str):   return x
    if isinstance(x, dict):  return x.get("value", "")
    if isinstance(x, list):  return "\n".join(m.get("value","") for m in x if isinstance(m,dict))
    return str(x)


def load_dpo_dataset(n=None):
    raw = load_dataset(DATASET_NAME, split="train")
    records = []
    for rec in raw:
        if n is not None and len(records) >= n:
            break
        p = rec.get("prompt",""); c = rec.get("chosen",""); r = rec.get("rejected","")
        if isinstance(p, list):
            p = "\n".join(m.get("value","") for m in p
                          if isinstance(m,dict) and m.get("from","").lower()!="assistant").strip()
        else:
            p = _msg_to_text(p).strip()
        c = _msg_to_text(c).strip(); r = _msg_to_text(r).strip()
        if c and r:
            records.append({"prompt": p, "chosen": c, "rejected": r})
    from datasets import Dataset
    print(f"[Dataset] {len(records)} DPO pairs loaded.")
    return Dataset.from_list(records)


def dpo_collator(examples, tokenizer):
    # DPOTrainer pre-tokenizes the dataset before the collator sees it.
    # Examples will have *_input_ids / *_attention_mask / *_labels keys.
    if "chosen_input_ids" in examples[0]:
        from torch.nn.utils.rnn import pad_sequence
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        def pad_key(key, pad_val):
            seqs = [torch.tensor(e[key], dtype=torch.long) for e in examples]
            return pad_sequence(seqs, batch_first=True, padding_value=pad_val)

        batch = {}
        for key in examples[0].keys():
            if "input_ids" in key:
                batch[key] = pad_key(key, pad_id)
            elif "attention_mask" in key:
                batch[key] = pad_key(key, 0)
            elif "labels" in key:
                batch[key] = pad_key(key, -100)
            else:
                batch[key] = [e[key] for e in examples]
        return batch

    # Fallback: raw text strings (legacy path)
    prompts  = [e["prompt"]   for e in examples]
    chosens  = [e["chosen"]   for e in examples]
    rejects  = [e["rejected"] for e in examples]
    def enc(texts, maxlen):
        return tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=maxlen, pad_to_multiple_of=8)
    ep = enc(prompts, 512); ec = enc(chosens, MAX_LENGTH); er = enc(rejects, MAX_LENGTH)
    for k in ("input_ids","attention_mask"):
        ep[k]=ep[k].long(); ec[k]=ec[k].long(); er[k]=er[k].long()
    return {
        "prompt_input_ids": ep["input_ids"], "prompt_attention_mask": ep["attention_mask"],
        "chosen_input_ids": ec["input_ids"], "chosen_attention_mask": ec["attention_mask"],
        "rejected_input_ids": er["input_ids"],"rejected_attention_mask": er["attention_mask"],
    }


# ============================================================
# Loss-capture callback
# ============================================================

class LossCaptureCallback(TrainerCallback):
    """Records DPO loss at every step."""
    def __init__(self, label):
        self.label  = label
        self.losses = []   # [(step, loss)]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append((state.global_step, logs["loss"]))


# ============================================================
# Cold-start mask generation
# ============================================================

def generate_cold_start_masks(model, tokenizer, device, cal_dataset):
    """Generate CAV and SNIP masks from calibration data (no training)."""
    chosen_texts   = [r["prompt"] + "\n" + r["chosen"]   for r in cal_dataset]
    rejected_texts = [r["prompt"] + "\n" + r["rejected"] for r in cal_dataset]

    os.makedirs("masks", exist_ok=True)

    # --- CAV ---
    print("\n" + "="*60)
    print("GENERATING CAV MASK (inference only)")
    print("="*60)
    extractor = FeatureExtractor()
    extractor.register(model)
    pos_acts = extractor.collect(model, tokenizer, chosen_texts,   device, max_length=MAX_LENGTH, batch_size=4)
    neg_acts = extractor.collect(model, tokenizer, rejected_texts, device, max_length=MAX_LENGTH, batch_size=4)
    extractor.remove()

    cav_scorer    = CAVProbeScorer()
    neuron_scores = cav_scorer.score(pos_acts, neg_acts)
    cav_masks     = cav_scorer.scores_to_masks(neuron_scores, model, sparsity_percent=SPARSITY)
    torch.save({"masks": cav_masks, "metadata": {"method":"cav","sparsity":SPARSITY}}, COLD_MASK_CAV)
    print(f"[CAV] Mask saved → {COLD_MASK_CAV}")

    # --- SNIP ---
    print("\n" + "="*60)
    print("GENERATING SNIP MASK (single backward pass, no weight update)")
    print("="*60)
    snip_scorer  = SNIPScorer()
    snip_scores  = snip_scorer.score(model, tokenizer, chosen_texts, device, max_length=MAX_LENGTH, batch_size=4)
    snip_masks   = snip_scorer.scores_to_masks(snip_scores, sparsity_percent=SPARSITY)
    torch.save({"masks": snip_masks, "metadata": {"method":"snip","sparsity":SPARSITY}}, COLD_MASK_SNIP)
    print(f"[SNIP] Mask saved → {COLD_MASK_SNIP}")

    return cav_masks, snip_masks


# ============================================================
# Single training run
# ============================================================

def run_dpo(label, model, tokenizer, dpo_dataset, device,
            mask_manager=None, n_steps=N_TRAIN_STEPS):
    """
    Run DPO training for n_steps. Returns list of (step, loss).
    mask_manager=None → dense training (standard AdamW).
    """
    print(f"\n{'='*60}\nRUN: {label}\n{'='*60}")

    run_dir = os.path.join(RESULTS_DIR, label.replace(" ", "_"))
    log_path = os.path.join(run_dir, "losses.csv")

    # ── Resume: skip this run if CSV already exists ──────────────────
    if os.path.exists(log_path):
        print(f"\n[{label}] Skipping — results already exist at {log_path}")
        losses = []
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                losses.append((int(row["step"]), float(row["loss"])))
        return losses

    os.makedirs(run_dir, exist_ok=True)

    cfg = DPOConfig(
        output_dir=os.path.join(run_dir, "ckpt"),
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        max_steps=n_steps,
        logging_steps=1,
        report_to="none",
        beta=0.1,
        max_length=MAX_LENGTH,
        max_prompt_length=512,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        save_steps=9999,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dpo_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    # Wrap the collator to guarantee input_ids / labels are Long tensors.
    # TRL ≥0.10 sometimes stores tokenized ids as float32 in the dataset.
    _base_collator = trainer.data_collator
    def _safe_collator(features):
        batch = _base_collator(features)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and ("input_ids" in k or "labels" in k):
                batch[k] = v.long()
        return batch
    trainer.data_collator = _safe_collator

    # Attach sparse optimizer when mask is provided
    if mask_manager is not None:
        from src.optimizers.sparse_adamw import SparseAdamW
        optimizer = SparseAdamW(
            list(model.named_parameters()),
            mask_manager,
            lr=LR,
            mlp_only=True,
        )
        trainer.optimizer = optimizer

    loss_cb = LossCaptureCallback(label)
    trainer.add_callback(loss_cb)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save loss log
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        w.writerows(loss_cb.losses)

    print(f"[{label}] {n_steps} steps in {elapsed:.1f}s | "
          f"final loss={loss_cb.losses[-1][1]:.4f}" if loss_cb.losses else "no loss logged")

    return loss_cb.losses


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- Shared resources ----
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading dataset...")
    dpo_dataset = load_dpo_dataset(SUBSET_SIZE)
    cal_dataset = [dpo_dataset[i] for i in range(min(N_CAL_SAMPLES, len(dpo_dataset)))]

    def fresh_model():
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        m.config.use_cache = False
        return m

    # ================================================================
    # Step 1 – Generate cold-start masks (uses a fresh model, no train)
    # ================================================================
    if os.path.exists(COLD_MASK_CAV) and os.path.exists(COLD_MASK_SNIP):
        print(f"\n[Masks] Found existing masks — skipping generation.")
        print(f"  CAV : {COLD_MASK_CAV}")
        print(f"  SNIP: {COLD_MASK_SNIP}")
        cav_masks  = torch.load(COLD_MASK_CAV,  map_location="cpu")["masks"]
        snip_masks = torch.load(COLD_MASK_SNIP, map_location="cpu")["masks"]
    else:
        print("\nLoading model for mask generation...")
        mask_model = fresh_model()
        cav_masks, snip_masks = generate_cold_start_masks(mask_model, tokenizer, device, cal_dataset)
        del mask_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ================================================================
    # Step 2 – Dense baseline (standard AdamW, no mask)
    # ================================================================
    dense_model = fresh_model()
    dense_losses = run_dpo("dense_baseline", dense_model, tokenizer, dpo_dataset, device, mask_manager=None)
    del dense_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ================================================================
    # Step 3 – Sparse DPO with CAV cold-start mask
    # ================================================================
    cav_model = fresh_model()
    cav_mm    = SparseMaskManager(COLD_MASK_CAV, device=str(device))
    cav_losses = run_dpo("cold_start_CAV", cav_model, tokenizer, dpo_dataset, device, mask_manager=cav_mm)
    del cav_model, cav_mm
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ================================================================
    # Step 4 – Sparse DPO with SNIP cold-start mask
    # ================================================================
    snip_model = fresh_model()
    snip_mm    = SparseMaskManager(COLD_MASK_SNIP, device=str(device))
    snip_losses = run_dpo("cold_start_SNIP", snip_model, tokenizer, dpo_dataset, device, mask_manager=snip_mm)
    del snip_model, snip_mm
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ================================================================
    # Step 5 – Sparse DPO with warm-start mask (if it exists)
    # ================================================================
    warm_losses = []
    if os.path.exists(WARM_MASK_PATH):
        warm_model = fresh_model()
        warm_mm    = SparseMaskManager(WARM_MASK_PATH, device=str(device))
        warm_losses = run_dpo("warm_start_momentum", warm_model, tokenizer, dpo_dataset, device, mask_manager=warm_mm)
        del warm_model, warm_mm
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print(f"\n[Warm-start] Skipping — mask not found at {WARM_MASK_PATH}")

    # ================================================================
    # Results
    # ================================================================
    print("\n" + "="*60)
    print("RESULTS – DPO Loss over 10 Steps")
    print("="*60)

    header = f"{'Step':>5}  {'Dense':>10}  {'CAV (cold)':>12}  {'SNIP (cold)':>13}  {'Warm-start':>12}"
    print(header)
    print("-" * len(header))

    def get(losses, step):
        d = dict(losses)
        return f"{d[step]:.4f}" if step in d else "   N/A"

    all_steps = sorted(set(s for s,_ in dense_losses + cav_losses + snip_losses + warm_losses))
    for step in all_steps:
        print(f"{step:>5}  {get(dense_losses,step):>10}  "
              f"{get(cav_losses,step):>12}  "
              f"{get(snip_losses,step):>13}  "
              f"{get(warm_losses,step):>12}")

    # Summary
    def final(losses):
        return losses[-1][1] if losses else float("nan")

    summary = {
        "dense_baseline":     final(dense_losses),
        "cold_start_CAV":     final(cav_losses),
        "cold_start_SNIP":    final(snip_losses),
        "warm_start_momentum":final(warm_losses),
    }
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Summary] Final losses after {N_TRAIN_STEPS} steps:")
    for name, loss in summary.items():
        tag = ""
        if name != "dense_baseline" and not (loss != loss):  # not nan
            delta = loss - summary["dense_baseline"]
            tag = f"  (Δ vs dense: {delta:+.4f})"
        print(f"  {name:<28}: {loss:.4f}{tag}")

    print(f"\nFull logs → {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()


import torch
from datasets import load_dataset
from typing import List, Dict, Any

def load_dpo_dataset(dataset_name, subset_size=None, split="train"):
    """Load and normalize DPO dataset from HuggingFace."""
    print(f"Loading dataset: {dataset_name}")
    raw_ds = load_dataset(dataset_name, split=split)
    
    def msg_to_text(x):
        """Robustly convert any DPO field value to plain text."""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            # Try all common message-content key names
            for key in ("value", "content", "text", "message"):
                if key in x:
                    return str(x[key])
            # Last resort: join all string values
            return " ".join(str(v) for v in x.values() if isinstance(v, str))
        if isinstance(x, list):
            parts = []
            for m in x:
                if isinstance(m, dict):
                    for key in ("value", "content", "text", "message"):
                        if key in m:
                            parts.append(str(m[key]))
                            break
                elif isinstance(m, str):
                    parts.append(m)
            return "\n".join(parts)
        return str(x)

    def extract_prompt(rec):
        """Extract the human/user prompt from conversations or prompt field."""
        convs = rec.get("conversations", None)
        if convs and isinstance(convs, list):
            human_parts = [
                m["value"] for m in convs
                if isinstance(m, dict)
                and m.get("from", m.get("role", "")).lower() in ("human", "user", "system")
                and "value" in m
            ]
            if human_parts:
                return "\n".join(human_parts).strip()
        
        # Fallback to 'prompt' field
        prompt_raw = rec.get("prompt", "")
        if isinstance(prompt_raw, list):
            prompt_text = "\n".join(
                m.get("value","") for m in prompt_raw
                if isinstance(m, dict) and m.get("from","").lower() != "assistant"
            ).strip()
            if prompt_text: return prompt_text
        
        return msg_to_text(prompt_raw).strip()

    def normalize_record(rec):
        prompt_text = extract_prompt(rec)
        chosen_text = msg_to_text(rec.get("chosen", "")).strip()
        rejected_text = msg_to_text(rec.get("rejected", "")).strip()

        return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}

    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    
    if subset_size is not None:
        norm_ds = norm_ds.select(range(min(subset_size, len(norm_ds))))
    
    print(f"✓ Loaded {len(norm_ds)} examples")
    return norm_ds


def dpo_collator_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """Data collator for DPO training."""
    # If the dataset was already preprocessed/tokenized (unlikely in this flow but supported)
    if "prompt_input_ids" in examples[0]:
        def pad_stack(key):
            seqs = [torch.tensor(ex[key]) if not torch.is_tensor(ex[key]) else ex[key] for ex in examples]
            lens = [s.size(-1) for s in seqs]
            maxlen = max(lens)
            out = torch.full((len(seqs), maxlen), fill_value=0, dtype=torch.long)
            mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.size(-1)] = s.to(torch.long)
                mask[i, : s.size(-1)] = 1
            return out, mask

        p_ids, p_mask = pad_stack("prompt_input_ids")
        c_ids, c_mask = pad_stack("chosen_input_ids")
        r_ids, r_mask = pad_stack("rejected_input_ids")
        return {
            "prompt_input_ids": p_ids, "prompt_attention_mask": p_mask,
            "chosen_input_ids": c_ids, "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids, "rejected_attention_mask": r_mask,
        }

    # Standard flow: raw strings to tokenized batches
    prompts  = [ex.get("prompt", "")   for ex in examples]
    chosens  = [ex.get("chosen", "")   for ex in examples]
    rejects  = [ex.get("rejected", "") for ex in examples]

    enc_prompt = [tokenizer(p, truncation=True, max_length=512) for p in prompts]
    enc_chosen = [tokenizer(c, truncation=True, max_length=1024) for c in chosens]
    enc_reject = [tokenizer(r, truncation=True, max_length=1024) for r in rejects]

    batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)
    batch_chosen = tokenizer.pad(enc_chosen, padding=True, return_tensors="pt", pad_to_multiple_of=8)
    batch_reject = tokenizer.pad(enc_reject, padding=True, return_tensors="pt", pad_to_multiple_of=8)

    for k in ("input_ids", "attention_mask"):
        batch_prompt[k] = batch_prompt[k].to(torch.long)
        batch_chosen[k] = batch_chosen[k].to(torch.long)
        batch_reject[k] = batch_reject[k].to(torch.long)

    return {
        "prompt_input_ids":        batch_prompt["input_ids"],
        "prompt_attention_mask":   batch_prompt["attention_mask"],
        "chosen_input_ids":        batch_chosen["input_ids"],
        "chosen_attention_mask":   batch_chosen["attention_mask"],
        "rejected_input_ids":      batch_reject["input_ids"],
        "rejected_attention_mask": batch_reject["attention_mask"],
    }


def concatenated_dpo_collator_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """
    Collator that concatenates prompt + chosen and prompt + rejected.
    Returns dictionaries containing combined input_ids and the exact prompt_len
    for selective activation pooling/masking.
    """
    prompts  = [ex.get("prompt", "")   for ex in examples]
    chosens  = [ex.get("chosen", "")   for ex in examples]
    rejects  = [ex.get("rejected", "") for ex in examples]

    def encode_and_mask(p_list, r_list):
        full_ids = []
        full_masks = []
        p_lens = []
        
        for p, r in zip(p_list, r_list):
            # Tokenize prompt to get its length
            p_enc = tokenizer(p, truncation=True, max_length=512)
            # trl-style: prompt + chosen (sometimes with a space)
            # We use a space for consistency with common DPO practices
            full_enc = tokenizer(p + " " + r, truncation=True, max_length=1024)
            
            full_ids.append(torch.tensor(full_enc["input_ids"]))
            full_masks.append(torch.tensor(full_enc["attention_mask"]))
            p_lens.append(len(p_enc["input_ids"]))
            
        # Pad sequences
        def pad_batch(tensor_list, pad_val=0):
            maxlen = max(t.size(0) for t in tensor_list)
            out = torch.full((len(tensor_list), maxlen), fill_value=pad_val, dtype=torch.long)
            for i, t in enumerate(tensor_list):
                out[i, :t.size(0)] = t
            return out

        return (
            pad_batch(full_ids, tokenizer.pad_token_id), 
            pad_batch(full_masks, 0), 
            torch.tensor(p_lens)
        )

    c_ids, c_mask, c_plens = encode_and_mask(prompts, chosens)
    r_ids, r_mask, r_plens = encode_and_mask(prompts, rejects)

    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_prompt_lens": c_plens,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_prompt_lens": r_plens,
    }

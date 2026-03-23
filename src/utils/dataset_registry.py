"""
Dataset Registry for Multi-Domain DPO Training

Central registry mapping short keys to HuggingFace dataset configs.
Handles per-dataset field remapping so that all datasets present a
uniform {prompt, chosen, rejected} interface for DPO training.

Usage:
    from src.utils.dataset_registry import get_dataset_config, load_dpo_dataset

    config = get_dataset_config("math-step-dpo")
    dataset = load_dpo_dataset("math-step-dpo", subset_size=100)
"""

from datasets import load_dataset
from typing import Optional, Dict, Any, List


# =============================================================================
# Registry Definition
# =============================================================================

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "light-r1": {
        "hf_id": "qihoo360/Light-R1-DPOData",
        "domain": "math_reasoning",
        "description": "Light-R1 DPO data (Math/Reasoning) — original default",
        "sanitized_name": "light_r1",
        "field_map": None,  # Uses standard prompt/chosen/rejected via data_utils
    },
    "tulu3": {
        "hf_id": "allenai/llama-3.1-tulu-3-8b-preference-mixture",
        "domain": "instruction_following",
        "description": "Tulu 3 8B preference mixture (Instruction Following)",
        "sanitized_name": "tulu3",
        "field_map": None,  # chosen/rejected are list-of-dicts, handled by data_utils.msg_to_text
    },
    "math-step-dpo": {
        "hf_id": "xinlai/Math-Step-DPO-10K",
        "domain": "math",
        "description": "Step-DPO 10K (Math step-by-step reasoning)",
        "sanitized_name": "math_step_dpo",
        "field_map": None,  # Has standard prompt/chosen/rejected fields
    },
    "codepref": {
        "hf_id": "Vezora/Code-Preference-Pairs",
        "domain": "coding",
        "description": "Code Preference Pairs (Coding bug-fix preference)",
        "sanitized_name": "codepref",
        "field_map": {
            # CodePref JSONL may use different field names; we handle common variants.
            # The existing data_utils normalizer handles most cases, but we add
            # explicit remapping here as a safety net.
            "accepted": "chosen",
            "rejected": "rejected",  # identity
        },
    },
}


def list_datasets() -> List[str]:
    """Return all registered dataset keys."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_config(key: str) -> Dict[str, Any]:
    """
    Look up a dataset config by registry key.
    
    Also accepts raw HuggingFace dataset IDs (e.g. "qihoo360/Light-R1-DPOData")
    for backward compatibility — returns a minimal config dict.
    """
    if key in DATASET_REGISTRY:
        return DATASET_REGISTRY[key]
    
    # Check if key matches any hf_id in the registry
    for rkey, cfg in DATASET_REGISTRY.items():
        if cfg["hf_id"] == key:
            return cfg
    
    # Unknown dataset — return a passthrough config for backward compat
    sanitized = key.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    
    return {
        "hf_id": key,
        "domain": "unknown",
        "description": f"Custom dataset: {key}",
        "sanitized_name": sanitized,
        "field_map": None,
    }


def sanitize_dataset_name(key: str) -> str:
    """Get a filesystem-safe name for a dataset key."""
    return get_dataset_config(key)["sanitized_name"]


def _apply_field_map(dataset, field_map: Dict[str, str]):
    """
    Rename columns in a HuggingFace dataset according to field_map.
    field_map maps source_column -> target_column.
    Only renames if the source column exists and differs from the target.
    """
    cols = dataset.column_names
    for src, tgt in field_map.items():
        if src in cols and src != tgt:
            dataset = dataset.rename_column(src, tgt)
    return dataset


def load_dpo_dataset(key_or_hf_id: str, subset_size: Optional[int] = None, split: str = "train"):
    """
    Load and normalize a DPO dataset, resolving registry keys.
    
    Leverages the existing robust normalizer in src.utils.data_utils
    after applying any dataset-specific field remapping.
    
    Args:
        key_or_hf_id: Registry key (e.g. "tulu3") or HuggingFace ID
        subset_size: Optional limit on number of examples
        split: Dataset split to load
        
    Returns:
        Normalized HuggingFace Dataset with {prompt, chosen, rejected} fields
    """
    from src.utils.data_utils import load_dpo_dataset as _base_load_dpo
    
    config = get_dataset_config(key_or_hf_id)
    hf_id = config["hf_id"]
    field_map = config.get("field_map")
    
    print(f"[dataset_registry] Resolved '{key_or_hf_id}' → {hf_id} (domain: {config['domain']})")
    
    if field_map:
        # Load raw first, apply field remapping, then normalize
        print(f"[dataset_registry] Applying field remapping: {field_map}")
        raw_ds = load_dataset(hf_id, split=split)
        raw_ds = _apply_field_map(raw_ds, field_map)
        
        # Now call the normalizer on the remapped dataset
        # We pass the hf_id but the dataset is already loaded, so we
        # re-implement the normalization inline using data_utils patterns
        from src.utils.data_utils import load_dpo_dataset as _load_raw
        
        # The base loader expects a dataset name string and re-downloads.
        # Since we already have a remapped dataset, we normalize it directly.
        return _normalize_dataset(raw_ds, subset_size=subset_size, label=hf_id)
    else:
        # Standard path: let data_utils handle everything
        return _base_load_dpo(hf_id, subset_size=subset_size, split=split)


def _normalize_dataset(raw_ds, subset_size=None, label="dataset"):
    """
    Normalize a pre-loaded dataset to {prompt, chosen, rejected} format.
    
    Mirrors the logic in data_utils.load_dpo_dataset but works on an
    already-loaded Dataset object.
    """
    def _msg_to_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for key in ("value", "content", "text", "message"):
                if key in x:
                    return str(x[key])
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

    def _extract_prompt(rec):
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
        
        prompt_raw = rec.get("prompt", "")
        if isinstance(prompt_raw, list):
            prompt_text = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
            if prompt_text:
                return prompt_text
        
        return _msg_to_text(prompt_raw).strip()

    def normalize_record(rec):
        prompt_text = _extract_prompt(rec)
        chosen_text = _msg_to_text(rec.get("chosen", "")).strip()
        rejected_text = _msg_to_text(rec.get("rejected", "")).strip()
        return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}

    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    
    if subset_size is not None:
        norm_ds = norm_ds.select(range(min(subset_size, len(norm_ds))))
    
    print(f"✓ Loaded {len(norm_ds)} examples from {label}")
    return norm_ds

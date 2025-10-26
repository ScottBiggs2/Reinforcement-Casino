# src/data/load_openr1.py
from datasets import load_dataset
from transformers import AutoTokenizer
import random

INPUT_CANDIDATES = ["problem", "query", "prompt", "input", "text"]
OUTPUT_CANDIDATES = ["solution", "response", "completion", "answer", "output"]

def find_field(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def load_openr1_subset(tokenizer_name="google/gemma-3-270m",
                       split="train",
                       subset_size=2000,
                       seed=42,
                       max_length=512):
    """
    Load and tokenize a subset of the OpenR1-Math-220k dataset.

    Returns:
        dataset (datasets.Dataset): tokenized subset with columns:
            - prompt (str)
            - label (str)
            - input_ids (list[int] per example)
            - attention_mask (list[int] per example)
        tokenizer
    """
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split=split) # streaming = True

    # Subsample deterministically
    if subset_size and subset_size < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(subset_size))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # detect input/output fields
    cols = dataset.column_names
    input_col = find_field(cols, INPUT_CANDIDATES)
    output_col = find_field(cols, OUTPUT_CANDIDATES)

    if input_col is None or output_col is None:
        raise KeyError(
            f"Could not find suitable input/output columns in dataset. "
            f"Available columns: {cols}. "
            f"Tried input candidates {INPUT_CANDIDATES} and output candidates {OUTPUT_CANDIDATES}."
        )

    def preprocess_batch(batch):
        # batch[input_col] is a list[str]
        toks = tokenizer(batch[input_col],
                         truncation=True,
                         padding="max_length",
                         max_length=max_length)
        return {
            "prompt": batch[input_col],
            "label": batch[output_col],
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }

    # Use batched mapping for speed and memory predictability
    dataset = dataset.map(preprocess_batch, batched=True, batch_size=128, remove_columns=[])
    return dataset, tokenizer

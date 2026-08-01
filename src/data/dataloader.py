# src/data/dataloader.py
from torch.utils.data import DataLoader
import torch
from .load_openr1 import load_openr1_subset

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    prompts = [item["prompt"] for item in batch]
    labels = [item["label"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": prompts,
        "labels": labels,
    }

def get_dataloader(tokenizer_name="google/gemma-3-270m",
                   subset_size=50,
                   batch_size=2,
                   shuffle=True):
    dataset, tokenizer = load_openr1_subset(
        tokenizer_name=tokenizer_name,
        subset_size=subset_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return dataloader, tokenizer

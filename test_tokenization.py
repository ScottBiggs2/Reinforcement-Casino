import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")

prompt_str = "What is 2+2?"
response_str = "It is 4."
full_str = prompt_str + "\n" + response_str

prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
full_ids = tokenizer.encode(full_str, add_special_tokens=True)

print("Prompt tokens:", tokenizer.convert_ids_to_tokens(prompt_ids))
print("Full tokens:", tokenizer.convert_ids_to_tokens(full_ids))
print("Prompt len:", len(prompt_ids), "Full len:", len(full_ids))

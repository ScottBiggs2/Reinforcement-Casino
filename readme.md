src
- cold_start
- full_training
- lora_training
- utils
- warm_start

# Be sure to use Python >= 3.13.5

# Remember to log in! 
```bash
wandb login [KEY HERE]
```
And for HuggingFace: 
```bash 
hf auth login [KEY HERE]
```

# So far, wont work on Explorer Tesla P100 GPUs... Ugh. 



```bash
python src/full_training/DPO_train.py
```


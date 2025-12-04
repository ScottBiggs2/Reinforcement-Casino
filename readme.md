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

# Must use H200 GPU

Full training run: 
```bash
python src/full_training/DPO_train.py
```

Mask finding (example kwargs): 
```bash
python src/warm_start/better_mask_finder.py --method momentum --top_k_percent 10.0 --target_step 25 --momentum_window 25
```

Triton Acceleration (example kwargs): 
```bash 
python src/sandbox/Triton_DPO_training_dev.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-100 \
  --mask masks/top_10.0pct_momentum_w25_step25.pt \
  --n_steps 10
```



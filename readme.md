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

Momentum (broken for now)
```bash
python src/warm_start/better_mask_finder.py --method momentum --top_k_percent 5.0 --target_step 25 --momentum_window 10
```

Magnitude: 
```bash 
python src/warm_start/better_mask_finder.py --method magnitude --top_k_percent 10.0 --target_step 25
```

Improvement and Jaccard scoring for masks: 

```bash
# Generate momentum mask at step 25, targeting 90% sparsity, with Jaccard metrics
python src/warm_start/even_better_mask_finder.py \
    --method momentum \
    --sparsity_percent 90.0 \
    --target_step 25 \
    --momentum_window 10 \
    --compute_jaccard
```

Triton Acceleration (example kwargs): 
```bash 
python src/sandbox/Triton_DPO_training_dev.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-100 \
  --mask masks/top_10.0pct_momentum_w25_step25.pt \
  --n_steps 10
```

Dev
```bash
python src/sandbox/Triton_DPO_training_dev.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-100 \
  --mask masks/top_10.0pct_magnitude_step25.pt \
  --n_steps 10
```

Main
```bash
python src/sandbox/Triton_DPO_training.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-100 \
  --mask masks/top_10.0pct_magnitude_step25.pt \
  --n_steps 10
```

Magic
```bash 
python src/magic/sparse_DPO_v2.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-100 \
  --mask masks/top_10.0pct_magnitude_step25.pt \
  --n_steps 10
```
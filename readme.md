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

Even Better Mask Finding:

```bash
# Generate momentum mask at step 25, targeting 90% sparsity, with Jaccard metrics
python src/warm_start/even_better_mask_finder.py \
    --method momentum \
    --sparsity_percent 90.0 \
    --target_step 25 \
    --momentum_window 10 \
    --compute_jaccard
```


```bash
# Test with magnitude first (simplest method)
python src/warm_start/even_better_mask_finder.py \
    --method magnitude \
    --sparsity_percent 90.0 \
    --target_step 25 \
    --compute_jaccard \
    --debug

# Or...
python src/warm_start/even_better_mask_finder.py \
    --method momentum \
    --sparsity_percent 90.0 \
    --target_step 100 \
    --momentum_window 80 \
    --compute_jaccard \
    --debug

# Or...
python src/warm_start/even_better_mask_finder.py \
    --method fisher \
    --sparsity_percent 90.0 \
    --target_step 100 \
    --compute_jaccard \
    --debug
```


Triton Acceleration (example kwargs): 

Magic
```bash 
python src/magic/sparse_DPO_v2.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-1000 \
  --mask masks/sparsity_90.0pct_magnitude_step100.pt \
  --n_steps 50
```

```bash 
python src/magic/sparse_DPO_v2.py \
  --checkpoint None \
  --mask masks/sparsity_90.0pct_magnitude_step100.pt \
  --n_steps 100
```


DPO Baseline timing:

```bash
python src/full_training/DPO_timing_baseline.py \
  --checkpoint None \
  --n_steps 50 \
  --subset_size 10 \ 
```
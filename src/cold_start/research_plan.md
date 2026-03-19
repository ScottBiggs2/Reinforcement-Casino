# Cold Start Plan

## Goal
Find a sparse task-specific mask without training, using only a small calibration set.

## Requirements
- Input: pretrained model + a few DPO samples
- Process: forward hooks or one backward pass, but no optimizer updates
- Output: `torch.save({"masks": masks, "metadata": ...})`

## Implemented direction

### CAV
- Compare chosen vs rejected activations with a linear probe
- Turn per-neuron scores into MLP weight masks
- Good when we want task-specific signal from DPO pairs

### SNIP
- Compute `|grad * weight|` from one loss backward pass
- Keep top-k important MLP weights directly
- Good as a simpler gradient-based baseline

## Files
- `cav/inference_mask_finder.py`: main entry point
- `utils/activation_hooks.py`: activation collection
- `utils/cav_probes.py`: CAV scoring
- `utils/snip_scorer.py`: SNIP scoring

## References
- CAV: Kim et al. (2018)
- SNIP: Lee et al. (2018)
- ROME: Meng et al. (2022)

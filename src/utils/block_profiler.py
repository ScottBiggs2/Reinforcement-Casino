import torch
from src.utils.slurm_safe_log import slurm_safe_print

def print_block_sparsity_profile(masks, block_size=16):
    """
    Analyzes a dictionary of binary masks to determine the 'real' block sparsity
    as seen by the GPU hardware (e.g., Triton BSR kernels).
    """
    total_elements = 0
    total_zeros = 0
    
    total_blocks = 0
    active_blocks = 0
    empty_blocks = 0
    mixed_blocks = 0
    dense_blocks = 0
    
    slurm_safe_print(f"\n{'='*60}")
    slurm_safe_print(f"BLOCK SPARSITY PROFILE (Block Size: {block_size}x{block_size})")
    slurm_safe_print(f"{'='*60}")
    
    for name, mask in masks.items():
        if mask.dim() != 2:
            continue
            
        M, N = mask.shape
        num_blocks_m = (M + block_size - 1) // block_size
        num_blocks_n = (N + block_size - 1) // block_size
        pad_m = num_blocks_m * block_size - M
        pad_n = num_blocks_n * block_size - N
        
        # Convert to float for padding (since bool padding is less predictable in some torch versions)
        if pad_m > 0 or pad_n > 0:
            padded = torch.nn.functional.pad(mask.float(), (0, pad_n, 0, pad_m), value=0)
            bool_mask = padded.bool()
        else:
            bool_mask = mask.bool()
            
        blocks = bool_mask.view(num_blocks_m, block_size, num_blocks_n, block_size)
        
        # Calculate block stats
        # blocks shape: (blocks_m, 16, blocks_n, 16)
        elements_per_block = blocks.sum(dim=(1, 3)) # Shape: (blocks_m, blocks_n)
        
        t_blocks = num_blocks_m * num_blocks_n
        a_blocks = (elements_per_block > 0).sum().item()
        e_blocks = (elements_per_block == 0).sum().item()
        d_blocks = (elements_per_block == (block_size * block_size)).sum().item()
        m_blocks = a_blocks - d_blocks
        
        total_blocks += t_blocks
        active_blocks += a_blocks
        empty_blocks += e_blocks
        mixed_blocks += m_blocks
        dense_blocks += d_blocks
        
        # Element stats
        total_elements += mask.numel()
        total_zeros += (~mask.bool()).sum().item()

    if total_elements == 0:
        slurm_safe_print("No valid 2D masks found.")
        return

    element_sparsity = (total_zeros / total_elements) * 100
    block_sparsity = (empty_blocks / total_blocks) * 100
    
    slurm_safe_print(f"Total Parameters:    {total_elements:,}")
    slurm_safe_print(f"Element Sparsity:    {element_sparsity:.4f}% (Theoretical compute skipped)")
    slurm_safe_print(f"")
    slurm_safe_print(f"Total Blocks:        {total_blocks:,}")
    slurm_safe_print(f"Empty Blocks:        {empty_blocks:,} (Actually skipped by GPU)")
    slurm_safe_print(f"Active Blocks:       {active_blocks:,} (Computed by GPU)")
    slurm_safe_print(f"  - Fully Dense:     {dense_blocks:,}")
    slurm_safe_print(f"  - Mixed (Waste):   {mixed_blocks:,}")
    slurm_safe_print(f"")
    slurm_safe_print(f"Effective Hardware Block Sparsity: {block_sparsity:.4f}%")
    
    if block_sparsity < element_sparsity - 10:
        slurm_safe_print(f"\nWARNING: Huge gap between Element ({element_sparsity:.1f}%) and Block ({block_sparsity:.1f}%) sparsity!")
        slurm_safe_print(f"The GPU is computing many zeros inside 'Mixed' blocks. Speedup will be lower than expected.")
    
    slurm_safe_print(f"{'='*60}\n")
    
    return {
        "element_sparsity": element_sparsity,
        "block_sparsity": block_sparsity,
        "total_blocks": total_blocks,
        "active_blocks": active_blocks,
        "empty_blocks": empty_blocks,
        "mixed_blocks": mixed_blocks,
        "dense_blocks": dense_blocks
    }

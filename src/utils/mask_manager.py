
import os
import torch

from src.utils.slurm_safe_log import slurm_safe_print

class SparseMaskManager:
    """
    Manages loading and applying sparse masks to model parameters.
    
    OPTIMIZATION: Pre-computes non-zero indices for each mask to enable
    true sparse operations (gather/scatter) instead of dense masking.
    """
    
    def __init__(self, mask_path, device='cuda'):
        slurm_safe_print(f"Loading masks from {mask_path}...")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        if mask_path.endswith('.json'):
            raise ValueError(f"Mask path points to a JSON file ({mask_path}). Please provide the .pt PyTorch file containing the actual masks.")

        try:
            # Try safe load first
            self.masks = torch.load(mask_path, map_location='cpu', weights_only=True)
        except Exception:
            # Fallback for older formats or complex dicts
            slurm_safe_print(f"Warning: safe load failed for {mask_path}, trying weights_only=False")
            self.masks = torch.load(mask_path, map_location='cpu', weights_only=False)
        self.device = device
        
        if isinstance(self.masks, dict) and 'masks' in self.masks:
            self.masks = self.masks['masks']
        
        slurm_safe_print(f"\nDEBUG - First 3 mask keys:")
        for i, key in enumerate(list(self.masks.keys())[:3]):
            slurm_safe_print(f"  {key}")

        # OPTIMIZATION: Pre-compute non-zero indices for each mask
        slurm_safe_print("\nPre-computing non-zero indices for sparse operations...")
        self.nonzero_indices = {}
        self.mask_stats = {}
        
        for name, mask in self.masks.items():
            # Binary masks only: bool storage (no fp16/bf32 mask tensors).
            if mask.dtype != torch.bool:
                mask = mask.ne(0).to(dtype=torch.bool)
            # Move mask to device
            mask = mask.to(device, non_blocking=True)
            self.masks[name] = mask

            # Compute statistics (bool: True = active weight)
            sparsity = (~mask).sum().item() / mask.numel()  # fraction False = pruned
            nonzero_count = mask.sum().item()
            
            self.mask_stats[name] = {
                'shape': tuple(mask.shape),
                'sparsity': sparsity,  # % zeros
                'nonzero': nonzero_count,  # count of non-zeros
            }
            
            # CRITICAL FIX: Pre-compute flattened indices of non-zero elements ON GPU
            # The mask is already on device, so nonzero() will return GPU indices
            if nonzero_count > 0:
                # 1. Unstructured indices (for fallback or unstructured kernels)
                flat_mask = mask.flatten()
                indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1).contiguous()
                self.nonzero_indices[name] = indices.to(device, non_blocking=True)
                
                # 2. Block-structured indices (for BSR kernels)
                # Assume 16x16 blocks (standard for this project)
                block_size = 16
                M, N = mask.shape
                num_blocks_m = (M + block_size - 1) // block_size
                num_blocks_n = (N + block_size - 1) // block_size
                pad_m = num_blocks_m * block_size - M
                pad_n = num_blocks_n * block_size - N
                
                if pad_m > 0 or pad_n > 0:
                    padded_mask = torch.nn.functional.pad(mask.float(), (0, pad_n, 0, pad_m), value=0)
                else:
                    padded_mask = mask.float()
                
                blocks = padded_mask.view(num_blocks_m, block_size, num_blocks_n, block_size)
                block_active = blocks.any(dim=1).any(dim=3) # any() over both block dims
                self.active_block_indices = getattr(self, 'active_block_indices', {})
                self.active_block_indices[name] = torch.nonzero(block_active.flatten(), as_tuple=False).squeeze(-1).to(torch.int32).to(device)
            else:
                self.nonzero_indices[name] = torch.empty(0, dtype=torch.long, device=device)
                self.active_block_indices = getattr(self, 'active_block_indices', {})
                self.active_block_indices[name] = torch.empty(0, dtype=torch.int32, device=device)
        
        slurm_safe_print(f"✓ Loaded {len(self.masks)} masks with pre-computed indices")
        self._print_mask_summary()
    
    def _print_mask_summary(self):
        """Print summary statistics of loaded masks."""
        total_params = sum(stats['shape'][0] * stats['shape'][1] 
                          for stats in self.mask_stats.values() 
                          if len(stats['shape']) == 2)
        total_active = sum(stats['nonzero'] 
                          for stats in self.mask_stats.values())
        
        # Calculate true sparsity (% of zeros)
        total_sparsity = 1.0 - (total_active / total_params) if total_params > 0 else 0
        
        slurm_safe_print(f"  Total parameters covered: {total_params:,}")
        slurm_safe_print(f"  Total active (non-zero) parameters: {total_active:,}")
        slurm_safe_print(f"  Overall sparsity (% zeros): {total_sparsity*100:.2f}%")
        slurm_safe_print(f"  Active parameters (% non-zero): {(1-total_sparsity)*100:.2f}%")
    
    def get_mask(self, param_name):
        """Get mask for a parameter - try direct match first, then with conversions."""
        if param_name in self.masks:
            return self.masks[param_name]
        
        mask_key = param_name.replace('.', '_')
        if mask_key in self.masks:
            return self.masks[mask_key]
        
        mask_key_dots = param_name.replace('_', '.')
        if mask_key_dots in self.masks:
            return self.masks[mask_key_dots]
        
        return None
    
    def get_nonzero_indices(self, param_name):
        """Get pre-computed non-zero indices for a parameter."""
        if param_name in self.nonzero_indices:
            return self.nonzero_indices[param_name]
        
        idx_key = param_name.replace('.', '_')
        if idx_key in self.nonzero_indices:
            return self.nonzero_indices[idx_key]
        
        idx_key_dots = param_name.replace('_', '.')
        if idx_key_dots in self.nonzero_indices:
            return self.nonzero_indices[idx_key_dots]
        
    def get_active_block_indices(self, param_name):
        """Get pre-computed active block indices (16x16) for a parameter."""
        if hasattr(self, 'active_block_indices'):
            if param_name in self.active_block_indices:
                return self.active_block_indices[param_name]
            
            idx_key = param_name.replace('.', '_')
            if idx_key in self.active_block_indices:
                return self.active_block_indices[idx_key]
            
            idx_key_dots = param_name.replace('_', '.')
            if idx_key_dots in self.active_block_indices:
                return self.active_block_indices[idx_key_dots]
        
        return None

    def is_mlp_layer(self, param_name):
        """Check if parameter belongs to an MLP layer."""
        return 'mlp' in param_name.lower()
    
    def has_mask(self, param_name):
        """Check if a mask exists for this parameter."""
        return (
            param_name in self.masks or
            param_name.replace('.', '_') in self.masks or
            param_name.replace('_', '.') in self.masks
        )

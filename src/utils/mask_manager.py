
import os
import torch

class SparseMaskManager:
    """
    Manages loading and applying sparse masks to model parameters.
    
    OPTIMIZATION: Pre-computes non-zero indices for each mask to enable
    true sparse operations (gather/scatter) instead of dense masking.
    """
    
    def __init__(self, mask_path, device='cuda'):
        print(f"Loading masks from {mask_path}...")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        self.masks = torch.load(mask_path, map_location='cpu')
        self.device = device
        
        if isinstance(self.masks, dict) and 'masks' in self.masks:
            self.masks = self.masks['masks']
        
        print(f"\nDEBUG - First 3 mask keys:")
        for i, key in enumerate(list(self.masks.keys())[:3]):
            print(f"  {key}")
        
        # OPTIMIZATION: Pre-compute non-zero indices for each mask
        print("\nPre-computing non-zero indices for sparse operations...")
        self.nonzero_indices = {}
        self.mask_stats = {}
        
        for name, mask in self.masks.items():
            # Move mask to device FIRST
            mask = mask.to(device, non_blocking=True)
            self.masks[name] = mask
            
            # Compute statistics
            sparsity = (mask == 0.0).sum().item() / mask.numel()  # % of zeros
            nonzero_count = (mask != 0.0).sum().item()
            
            self.mask_stats[name] = {
                'shape': tuple(mask.shape),
                'sparsity': sparsity,  # % zeros
                'nonzero': nonzero_count,  # count of non-zeros
            }
            
            # CRITICAL FIX: Pre-compute flattened indices of non-zero elements ON GPU
            # The mask is already on device, so nonzero() will return GPU indices
            if nonzero_count > 0:
                # nonzero() returns indices; we flatten and store as 1D tensor
                flat_mask = mask.flatten()
                indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1).contiguous()
                # Ensure indices are on the correct device (should already be, but verify)
                self.nonzero_indices[name] = indices.to(device, non_blocking=True)
            else:
                # Edge case: completely sparse mask
                self.nonzero_indices[name] = torch.empty(0, dtype=torch.long, device=device)
        
        print(f"✓ Loaded {len(self.masks)} masks with pre-computed indices")
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
        
        print(f"  Total parameters covered: {total_params:,}")
        print(f"  Total active (non-zero) parameters: {total_active:,}")
        print(f"  Overall sparsity (% zeros): {total_sparsity*100:.2f}%")
        print(f"  Active parameters (% non-zero): {(1-total_sparsity)*100:.2f}%")
    
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

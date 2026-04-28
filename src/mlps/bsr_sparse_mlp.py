
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.kernels.bsr_backward import sparse_grad_input_triton, sparse_weight_gradient_triton
from src.utils.slurm_safe_log import slurm_safe_print


def _grad_input_mode() -> str:
    return os.environ.get("RL_CASINO_BSR_GRAD_INPUT_MODE", "dense").strip().lower()

# ============================================================================
# AUTOGRAD FUNCTIONS
# ============================================================================

class SparseLinearFunction(torch.autograd.Function):
    """
    Custom autograd function for linear layer with sparse backward.
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, mask, active_blocks, block_size=16, use_tf32=False):
        ctx.save_for_backward(input, weight, mask, active_blocks)
        ctx.has_bias = bias is not None
        ctx.block_size = block_size
        ctx.use_tf32 = use_tf32
        
        # Standard forward pass
        # Note: We assume the weight is already masked (or we don't care about forward sparsity for correctness here,
        # usually forward uses dense matmul for efficiency unless highly sparse).
        # We use F.linear with the weight as-is.
        output = F.linear(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight, mask, active_blocks = ctx.saved_tensors
        block_size = ctx.block_size
        
        grad_input = grad_weight = grad_bias = None
        
        # Gradient w.r.t input
        if ctx.needs_input_grad[0]:
            mode = _grad_input_mode()
            if mode in ("sparse", "block", "bsr"):
                grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
                if not grad_output_flat.is_contiguous():
                    grad_output_flat = grad_output_flat.contiguous()
                grad_input_flat = sparse_grad_input_triton(
                    grad_output_flat,
                    weight,
                    mask,
                    active_blocks=active_blocks,
                    block_size=block_size,
                    use_tf32=ctx.use_tf32,
                )
                grad_input = grad_input_flat.reshape_as(input_tensor)
            else:
                grad_input = grad_output @ weight
            
        # SPARSE gradient w.r.t weight
        if ctx.needs_input_grad[1]:
            # Flatten inputs for the kernel (handles 3D->2D: [B, S, D] -> [B*S, D])
            # The kernel expects (batch_size, dim), so we merge batch and seq dims.
            grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
            input_tensor_flat = input_tensor.reshape(-1, input_tensor.shape[-1])

            # Ensure contiguous for Triton
            if not grad_output_flat.is_contiguous():
                grad_output_flat = grad_output_flat.contiguous()
            if not input_tensor_flat.is_contiguous():
                input_tensor_flat = input_tensor_flat.contiguous()
                
            grad_weight = sparse_weight_gradient_triton(
                grad_output_flat, input_tensor_flat, mask,
                active_blocks=active_blocks,
                block_size=block_size,
                use_tf32=ctx.use_tf32
            )
        
            
        # Bias gradient
        if ctx.has_bias and ctx.needs_input_grad[2]:
            # Sum over all dimensions except the last one (channel/feature dim)
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
            
        return grad_input, grad_weight, grad_bias, None, None, None, None


class SparseLinearLayer(nn.Linear):
    """
    Drop-in replacement for nn.Linear that uses SparseLinearFunction for backward pass.
    """
    def __init__(self, in_features, out_features, bias=True, mask=None, block_size=16, use_tf32=False):
        super().__init__(in_features, out_features, bias)
        self.block_size = block_size
        self.use_tf32 = use_tf32
        self.register_buffer('mask', None)
        self.register_buffer('active_blocks', None)
        if mask is not None:
            self.set_mask(mask)
        
    def set_mask(self, mask):
        if mask.shape != self.weight.shape:
            raise ValueError(f"Mask shape {mask.shape} mismatch with weight {self.weight.shape}")
        self.mask = mask.to(self.weight.device)
        
        # Precompute active blocks
        output_dim, input_dim = mask.shape
        num_blocks_m = (output_dim + self.block_size - 1) // self.block_size
        num_blocks_n = (input_dim + self.block_size - 1) // self.block_size
        pad_m = num_blocks_m * self.block_size - output_dim
        pad_n = num_blocks_n * self.block_size - input_dim
        
        mask_f = mask.float() if mask.dtype == torch.bool else mask
        padded_mask = torch.nn.functional.pad(mask_f, (0, pad_n, 0, pad_m), value=0)
        blocks = padded_mask.view(num_blocks_m, self.block_size, num_blocks_n, self.block_size)
        block_active = blocks.any(dim=1).any(dim=2)
        
        self.active_blocks = torch.nonzero(block_active.flatten(), as_tuple=True)[0].to(torch.int32).to(self.weight.device)
        
        # Ensure weight is zeroed where mask is zero (mathematical equivalence)
        with torch.no_grad():
            self.weight.data *= self.mask.to(self.weight.dtype)
        
    def forward(self, input):
        if self.mask is not None:
             return SparseLinearFunction.apply(input, self.weight, self.bias, self.mask, self.active_blocks, self.block_size, self.use_tf32)
        else:
            return F.linear(input, self.weight, self.bias)

def replace_linear_modules(model, mask_dict, block_size=16, use_tf32=False, verbose=True):
    """
    Recursively replace nn.Linear modules with SparseLinearLayer.
    
    Args:
        model: PyTorch model
        mask_dict: Dictionary mapping parameter names to masks
        block_size: Block size for sparse kernel
        use_tf32: Whether to use TF32 for higher precision accumulation in the Triton kernel
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = f"{name}.weight"
            
            # Try to find mask
            mask = None
            if weight_name in mask_dict:
                mask = mask_dict[weight_name]
            
            # If no direct match, try to fuzzy match (handling potential prefix mismatches)
            if mask is None:
                for key in mask_dict:
                     if key.endswith(weight_name) or weight_name.endswith(key):
                         mask = mask_dict[key]
                         break
                         
            if mask is not None:
                if verbose:
                    slurm_safe_print(f"Replacing {name} with SparseLinearLayer (Mask active: {mask.shape})")
                
                # Create replacement
                sparse_layer = SparseLinearLayer(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    mask=mask,
                    block_size=block_size,
                    use_tf32=use_tf32
                )
                
                # Copy weights/bias
                sparse_layer.weight.data = module.weight.data
                if module.bias is not None:
                    sparse_layer.bias.data = module.bias.data
                
                # Replace in parent
                # We need to find the parent module and the attribute name
                parts = name.split('.')
                parent = model
                if len(parts) > 1:
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                attr_name = parts[-1]
                
                setattr(parent, attr_name, sparse_layer)

def restore_linear_modules(model, verbose=True):
    """
    Recursively replace SparseLinearLayer modules back with standard nn.Linear.
    This allows for normal saving/loading with standard HuggingFace tools.
    """
    for name, module in model.named_modules():
        if isinstance(module, SparseLinearLayer):
            if verbose:
                slurm_safe_print(f"Restoring {name} to standard nn.Linear")
            
            # Create replacement
            dense_layer = nn.Linear(
                module.in_features,
                module.out_features,
                module.bias is not None
            )
            
            # Copy weights/bias
            dense_layer.weight.data = module.weight.data
            if module.bias is not None:
                dense_layer.bias.data = module.bias.data
            
            # Replace in parent
            parts = name.split('.')
            parent = model
            if len(parts) > 1:
                for part in parts[:-1]:
                    parent = getattr(parent, part)
            attr_name = parts[-1]
            
            setattr(parent, attr_name, dense_layer)

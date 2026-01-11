
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ============================================================================
# TRITON KERNELS
# ============================================================================

@triton.jit
def sparse_grad_weight_kernel(
    # Outputs
    grad_weight_ptr,
    # Inputs
    grad_output_ptr,  # (batch, output_dim)
    input_ptr,        # (batch, input_dim)
    mask_ptr,         # (output_dim, input_dim)
    # Dimensions
    batch_size, output_dim, input_dim,
    stride_go_batch, stride_go_out,
    stride_in_batch, stride_in_in,
    stride_gw_out, stride_gw_in,
    stride_m_out, stride_m_in,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sparse backward pass kernel for MLP weight gradients.
    
    Computes: grad_W[out, in] = sum_b (grad_output[b, out] * input[b, in])
    But ONLY for blocks where mask is not all zeros.
    
    IMPROVEMENT: Checks the ENTIRE mask block to ensure correctness with element-wise sparsity,
    and applies the mask to the result to zero out gradients for masked weights within active blocks.
    """
    pid = tl.program_id(0)
    
    # 2D grid over (output_dim, input_dim)
    num_blocks_in = tl.cdiv(input_dim, BLOCK_SIZE)
    block_out = pid // num_blocks_in
    block_in = pid % num_blocks_in
    
    out_start = block_out * BLOCK_SIZE
    in_start = block_in * BLOCK_SIZE
    
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)
    in_offsets = in_start + tl.arange(0, BLOCK_SIZE)
    
    # Create valid mask for boundary checks
    out_valid = (out_offsets < output_dim)
    in_valid = (in_offsets < input_dim)
    valid = out_valid[:, None] & in_valid[None, :]
    
    # Load mask block
    mask_offsets = out_offsets[:, None] * stride_m_out + in_offsets[None, :] * stride_m_in
    mask_block = tl.load(mask_ptr + mask_offsets, mask=valid, other=0.0)
    
    # Check if block is fully masked
    # We use max because masks are typically 0.0 or 1.0 (or non-zero)
    if tl.max(mask_block) == 0.0:
        # This block is masked - write zeros and early exit
        gw_offsets = out_offsets[:, None] * stride_gw_out + in_offsets[None, :] * stride_gw_in
        tl.store(grad_weight_ptr + gw_offsets, 0.0, mask=valid)
        return  # FAST EXIT
        
    # Block is active (at least partially) - compute gradient
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over batch dimension
    for b_start in range(0, batch_size, 32):
        b_offsets = b_start + tl.arange(0, 32)
        b_mask = b_offsets < batch_size
        
        # Load grad_output[b, out]
        go_offsets = b_offsets[:, None] * stride_go_batch + out_offsets[None, :] * stride_go_out
        go_mask = b_mask[:, None] & out_valid[None, :]
        grad_out = tl.load(grad_output_ptr + go_offsets, mask=go_mask, other=0.0)
        
        # Load input[b, in]
        in_offsets_2d = b_offsets[:, None] * stride_in_batch + in_offsets[None, :] * stride_in_in
        in_mask_load = b_mask[:, None] & in_valid[None, :]
        inp = tl.load(input_ptr + in_offsets_2d, mask=in_mask_load, other=0.0)
        
        # Accumulate: grad_W[out, in] += grad_output[b, out] * input[b, in]
        # grad_out is (32, BLOCK_SIZE), inp is (32, BLOCK_SIZE)
        # We want grad_out.T @ inp -> (BLOCK_SIZE, 32) @ (32, BLOCK_SIZE) -> (BLOCK_SIZE, BLOCK_SIZE)
        acc += tl.dot(tl.trans(grad_out), inp)
        
    # Apply mask to gradient to ensure element-wise sparsity is respected
    # This acts as `grad *= mask` logic, ensuring we don't update masked weights
    # even if they are in an active block.
    acc = acc * mask_block
    
    # Store result
    gw_offsets = out_offsets[:, None] * stride_gw_out + in_offsets[None, :] * stride_gw_in
    tl.store(grad_weight_ptr + gw_offsets, acc.to(grad_weight_ptr.dtype.element_ty), mask=valid)


def sparse_weight_gradient_triton(grad_output, input_tensor, mask, block_size=16):
    """
    Compute sparse weight gradient using Triton kernel.
    """
    batch_size, output_dim = grad_output.shape
    batch_size, input_dim = input_tensor.shape
    
    grad_weight = torch.empty(output_dim, input_dim, device=grad_output.device, dtype=grad_output.dtype)
    
    # Launch kernel
    num_blocks_out = triton.cdiv(output_dim, block_size)
    num_blocks_in = triton.cdiv(input_dim, block_size)
    grid = (num_blocks_out * num_blocks_in,)
    
    sparse_grad_weight_kernel[grid](
        grad_weight,
        grad_output, input_tensor, mask,
        batch_size, output_dim, input_dim,
        grad_output.stride(0), grad_output.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
        mask.stride(0), mask.stride(1),
        BLOCK_SIZE=block_size,
    )
    
    return grad_weight


# ============================================================================
# AUTOGRAD FUNCTIONS
# ============================================================================

class SparseLinearFunction(torch.autograd.Function):
    """
    Custom autograd function for linear layer with sparse backward.
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, mask, block_size=16):
        ctx.save_for_backward(input, weight, mask)
        ctx.has_bias = bias is not None
        ctx.block_size = block_size
        
        # Standard forward pass
        # Note: We assume the weight is already masked (or we don't care about forward sparsity for correctness here,
        # usually forward uses dense matmul for efficiency unless highly sparse).
        # We use F.linear with the weight as-is.
        output = F.linear(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight, mask = ctx.saved_tensors
        block_size = ctx.block_size
        
        grad_input = grad_weight = grad_bias = None
        
        # Gradient w.r.t input
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
            
        # SPARSE gradient w.r.t weight
        if ctx.needs_input_grad[1]:
            # Ensure contiguous for Triton
            if not grad_output.is_contiguous():
                grad_output = grad_output.contiguous()
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()
                
            grad_weight = sparse_weight_gradient_triton(
                grad_output, input_tensor, mask, 
                block_size=block_size
            )
            
        # Bias gradient
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias, None, None


class SparseLinearLayer(nn.Linear):
    """
    Drop-in replacement for nn.Linear that uses SparseLinearFunction for backward pass.
    """
    def __init__(self, in_features, out_features, bias=True, mask=None, block_size=16):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)
        self.block_size = block_size
        
    def set_mask(self, mask):
        if mask.shape != self.weight.shape:
            raise ValueError(f"Mask shape {mask.shape} mismatch with weight {self.weight.shape}")
        self.mask = mask.to(self.weight.device)
        
    def forward(self, input):
        if self.mask is not None:
             # Pass block_size to the custom function
             return SparseLinearFunction.apply(input, self.weight, self.bias, self.mask, self.block_size)
        else:
            return F.linear(input, self.weight, self.bias)

def replace_linear_modules(model, mask_dict, block_size=16, verbose=True):
    """
    Recursively replace nn.Linear modules with SparseLinearLayer.
    
    Args:
        model: PyTorch model
        mask_dict: Dictionary mapping parameter names to masks
        block_size: Block size for sparse kernel
    """
    # Build mapping of weight parameter IDs to masks to handle shared weights or complex accessing
    # But mask_dict usually maps "model.layers.0.mlp.gate_proj.weight" -> mask
    
    # Helper to traverse
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if we have a mask for this module's weight
            # We need the full parameter name.
            # This is hard because named_modules is local? 
            # No, model.named_modules() gives full names like "model.layers.0..."
            
            weight_name = f"{name}.weight"
            
            # Try to find mask
            mask = None
            if weight_name in mask_dict:
                mask = mask_dict[weight_name]
            
            # If no direct match, try to fuzzy match (handling potential prefix mismatches)
            if mask is None:
                # Basic check: sometimes keys have 'module.' prefix or lack it
                for key in mask_dict:
                     if key.endswith(weight_name) or weight_name.endswith(key):
                         mask = mask_dict[key]
                         break
                         
            if mask is not None:
                if verbose:
                    print(f"Replacing {name} with SparseLinearLayer (Mask active: {mask.shape})")
                
                # Create replacement
                sparse_layer = SparseLinearLayer(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    mask=mask,
                    block_size=block_size
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

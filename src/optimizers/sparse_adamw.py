
import torch
import time
import os
from src.utils.mask_manager import SparseMaskManager
from src.utils.slurm_safe_log import slurm_safe_print
from src.kernels.indexed_sparse_adam import (
    triton_indexed_sparse_adamw_step,
    triton_bsr_sparse_adamw_step,
    triton_bsr_sparse_adamw_step_2d,
)
from src.kernels.sparse_norm import compute_bsr_grad_norm_sq, compute_unstructured_grad_norm_sq

class SparseAdamW(torch.optim.Optimizer):
    """
    Custom AdamW optimizer with Triton-accelerated sparse updates.
    
    OPTIMIZATIONS:
    1. Pre-initialized optimizer states (no lazy init overhead)
    2. Indexed sparse kernel using gather/scatter operations
    3. Only processes non-zero mask elements (~10% with 90% sparsity)
    4. FIXED: Precompute bias corrections to avoid kernel recompilation
    """
    
    def __init__(
        self,
        named_params,
        mask_manager: SparseMaskManager,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        block_size=128,
        mlp_only=False,
        max_grad_norm=1.0,
        *,
        eager_state_init: bool = True,
    ):
        self.param_to_name = {}
        params = []
        for name, param in named_params:
            params.append(param)
            self.param_to_name[id(param)] = name
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.mask_manager = mask_manager
        self.block_size = block_size
        self.mlp_only = mlp_only
        self.max_grad_norm = max_grad_norm # Default clipping value for stability
        
        self.stats = {
            'sparse_steps': 0,
            'dense_steps': 0,
            'nan_warnings': [],
            'dense_fallbacks': 0,
            'dense_fallback_params': set(),
        }
        self.eager_state_init = eager_state_init
        self._warned_empty_indices = set()
        self._warned_shape_mismatch = set()
        self._warned_fallback = set()

        # Pre-allocating exp_avg/exp_avg_sq for every parameter spikes memory (2× model size on-device).
        # Lazy init matches PyTorch Adam: allocate on first step — slower first step, much lower peak RAM/VRAM.
        if self.eager_state_init:
            slurm_safe_print(f"\nPre-initializing optimizer states for all parameters...")
            init_start = time.time()
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        p_name = self.param_to_name.get(id(p))
                        self._init_adam_state_tensors(p, p_name)

            init_time = time.time() - init_start
            slurm_safe_print(f"✓ Optimizer states pre-initialized in {init_time:.2f}s")
        else:
            slurm_safe_print(
                "\nSparseAdamW: lazy optimizer state (allocate on first step per parameter; lower peak memory)"
            )

        slurm_safe_print(f"✓ SparseAdamW optimizer ready")
        slurm_safe_print(f"  MLP-only: {mlp_only}")
        slurm_safe_print(f"  Block size: {block_size}")
        slurm_safe_print(f"  Using indexed sparse kernels: TRUE")
        slurm_safe_print(f"  Kernel recompilation fix: APPLIED")
        slurm_safe_print(f"  Local Gradient Clipping enabled: max_norm={self.max_grad_norm}")

    def _init_adam_state_tensors(self, p: torch.Tensor, param_name: str = None) -> None:
        state = self.state[p]
        state["step"] = 0
        
        should_use_sparse = (
            param_name is not None and 
            self.mask_manager.has_mask(param_name) and 
            len(p.shape) == 2 and 
            (not self.mlp_only or self.mask_manager.is_mlp_layer(param_name))
        )
        
        if should_use_sparse:
            active_block_indices = self.mask_manager.get_active_block_indices(param_name)
            if active_block_indices is not None and len(active_block_indices) > 0:
                # Normalize once so kernel launch never needs to cast/copy/sync.
                if active_block_indices.dtype != torch.int32:
                    active_block_indices = active_block_indices.to(torch.int32)
                if not active_block_indices.is_contiguous():
                    active_block_indices = active_block_indices.contiguous()
                # Use Block-Sparse storage (BSR-compatible)
                # We store entire 16x16 blocks (including zeros) to enable fast coalesced kernels
                n_blocks = active_block_indices.shape[0]
                n_elements = n_blocks * 16 * 16
                state["exp_avg"] = torch.zeros(n_elements, device=p.device, dtype=p.dtype)
                state["exp_avg_sq"] = torch.zeros(n_elements, device=p.device, dtype=p.dtype)
                state["sparse_regime"] = "block"
                state["active_block_indices"] = active_block_indices
            else:
                # Fallback to Element-Sparse storage (Unstructured)
                nonzero_indices = self.mask_manager.get_nonzero_indices(param_name)
                n_nonzero = nonzero_indices.shape[0]
                state["exp_avg"] = torch.zeros(n_nonzero, device=p.device, dtype=p.dtype)
                state["exp_avg_sq"] = torch.zeros(n_nonzero, device=p.device, dtype=p.dtype)
                state["sparse_regime"] = "element"
                state["nonzero_indices"] = nonzero_indices
        else:
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _ensure_adam_state(self, p: torch.Tensor, param_name: str = None) -> None:
        if not self.eager_state_init:
            state = self.state[p]
            if "exp_avg" not in state:
                self._init_adam_state_tensors(p, param_name)

    @torch.no_grad()
    def _clip_grads_sparse(self):
        """
        Optimized gradient clipping that only looks at non-zero elements.
        Avoids redundant memory reads and synchronizations.
        """
        if not self.max_grad_norm or self.max_grad_norm <= 0:
            return
            
        total_norm_sq = torch.tensor(0.0, device='cuda')
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_name = self.param_to_name.get(id(p), None)
                
                # Check for mask
                if param_name and self.mask_manager.has_mask(param_name):
                    # Try BSR norm first
                    active_blocks = self.mask_manager.get_active_block_indices(param_name)
                    if active_blocks is not None:
                        total_norm_sq += compute_bsr_grad_norm_sq(p.grad, active_blocks)
                    else:
                        # Fallback to unstructured
                        indices = self.mask_manager.get_nonzero_indices(param_name)
                        total_norm_sq += compute_unstructured_grad_norm_sq(p.grad, indices)
                else:
                    # Dense parameter
                    total_norm_sq += torch.sum(p.grad * p.grad)
        
        total_norm = total_norm_sq.sqrt()
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        
        if clip_coef < 1.0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.mul_(clip_coef)
        
        return total_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step using Triton kernels."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Check for redundant clipping (if Trainer also has max_grad_norm)
        # We handle clipping ourselves using optimized sparse kernels.
        self._clip_grads_sparse()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_name = self.param_to_name.get(id(p), None)
                
                if param_name is None:
                    self._dense_step(p, group)
                    continue
                
                self._ensure_adam_state(p, param_name)
                
                should_use_sparse = (
                    self.mask_manager.has_mask(param_name) and
                    len(p.shape) == 2 and
                    (not self.mlp_only or self.mask_manager.is_mlp_layer(param_name))
                )
                
                if should_use_sparse:
                    self._sparse_step(p, param_name, group)
                else:
                    self._dense_step(p, group)
        
        return loss
    
    def _sparse_step(self, param, param_name, group):
        """
        Sparse update using indexed Triton kernel (PERFORMANCE FIXED).
        """
        grad = param.grad
        mask = self.mask_manager.get_mask(param_name)
        nonzero_indices = self.mask_manager.get_nonzero_indices(param_name)
        
        if mask is None or nonzero_indices is None:
            slurm_safe_print(f"WARNING: Missing mask or indices for {param_name}, falling back to dense")
            self._dense_step(param, group)
            return
        
        if mask.shape != param.shape:
            if param_name not in self._warned_shape_mismatch:
                slurm_safe_print(f"WARNING: Shape mismatch for {param_name}")
                slurm_safe_print(f"  Param: {param.shape}, Mask: {mask.shape}")
                slurm_safe_print(f"  Falling back to dense update")
                self._warned_shape_mismatch.add(param_name)
            self._dense_step(param, group)
            return

        if nonzero_indices.numel() == 0:
            if param_name not in self._warned_empty_indices:
                slurm_safe_print(f"WARNING: Mask has 0 active elements for {param_name}; dense fallback.")
                self._warned_empty_indices.add(param_name)
            self._dense_step(param, group)
            return
        
        state = self.state[param]
        state['step'] += 1
        
        try:
            active_block_indices = state.get("active_block_indices", None)
            use_block = active_block_indices is not None and active_block_indices.numel() > 0

            if use_block:
                # Kernel choice for block regime.
                # Default stays 1D block kernel unless explicitly overridden.
                kernel_mode = os.environ.get("RL_CASINO_ADAM_KERNEL", "block_1d").strip().lower()
                step_fn = triton_bsr_sparse_adamw_step
                if kernel_mode in ("block_2d", "bsr_2d", "2d"):
                    step_fn = triton_bsr_sparse_adamw_step_2d

                step_fn(
                    param=param,
                    grad=grad,
                    active_blocks=active_block_indices,
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    lr=group["lr"],
                    beta1=group["betas"][0],
                    beta2=group["betas"][1],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    step=state["step"],
                    block_size=16,  # 16x16 BSR
                )
            else:
                # Unstructured indexed sparse kernel. Do not rely on state['nonzero_indices'].
                # If we want caching for debug visibility, store it when absent.
                if "nonzero_indices" not in state:
                    state["nonzero_indices"] = nonzero_indices

                triton_indexed_sparse_adamw_step(
                    param=param,
                    grad=grad,
                    nonzero_indices=nonzero_indices,
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    lr=group["lr"],
                    beta1=group["betas"][0],
                    beta2=group["betas"][1],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    step=state["step"],
                    block_size=self.block_size,
                )
        except Exception as e:
            slurm_safe_print(f"ERROR in indexed Triton kernel for {param_name}: {e}")
            slurm_safe_print(f"  Falling back to dense update")
            self.stats["dense_fallbacks"] += 1
            self.stats["dense_fallback_params"].add(param_name)
            if param_name not in self._warned_fallback:
                slurm_safe_print(
                    "  NOTE: dense fallback requires dense-shaped optimizer state; "
                    "we will reinitialize exp_avg/exp_avg_sq if they are sparse-shaped."
                )
                self._warned_fallback.add(param_name)
            self._dense_step(param, group)
            return
        
        self.stats['sparse_steps'] += 1
    
    def _dense_step(self, param, group):
        """Standard dense AdamW update (fallback for non-masked params)."""
        grad = param.grad
        state = self.state[param]
        state['step'] += 1

        # If this param was initialized with sparse-shaped optimizer state (1D packed),
        # make the fallback safe by reinitializing to dense tensors.
        exp_avg = state.get("exp_avg", None)
        exp_avg_sq = state.get("exp_avg_sq", None)
        if (
            exp_avg is None
            or exp_avg_sq is None
            or exp_avg.shape != param.shape
            or exp_avg_sq.shape != param.shape
        ):
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        
        beta1, beta2 = group['betas']
        
        # AdamW weight decay
        param.mul_(1 - group['lr'] * group['weight_decay'])
        
        # Update biased first and second moment estimates
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        # Compute step
        step_size = group['lr'] / bias_correction1
        denom = (state['exp_avg_sq'].sqrt() / bias_correction2 ** 0.5).add_(group['eps'])
        param.addcdiv_(state['exp_avg'], denom, value=-step_size)
        
        self.stats['dense_steps'] += 1
    
    def check_for_nans(self, model):
        """Check all parameters for NaN/Inf after training completes."""
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                nan_params.append((name, nan_count, inf_count))
        
        if nan_params:
            slurm_safe_print(f"\n{'='*60}")
            slurm_safe_print("WARNING: NaN/Inf DETECTED IN PARAMETERS")
            slurm_safe_print(f"{'='*60}")
            for name, nan_count, inf_count in nan_params:
                slurm_safe_print(f"  {name}: NaN={nan_count}, Inf={inf_count}")
            slurm_safe_print(f"{'='*60}\n")
            self.stats['nan_warnings'] = nan_params
        else:
            slurm_safe_print("\n✓ No NaN/Inf detected in parameters\n")
    
    def print_stats(self):
        """Print optimizer statistics."""
        total = self.stats['sparse_steps'] + self.stats['dense_steps']
        sparse = self.stats['sparse_steps']
        dense = self.stats['dense_steps']
        
        slurm_safe_print(f"\n{'='*60}")
        slurm_safe_print(f"SPARSE ADAMW OPTIMIZER STATISTICS")
        slurm_safe_print(f"{'='*60}")
        slurm_safe_print(f"Total steps:      {total:,}")
        slurm_safe_print(f"Sparse steps:     {sparse:,} ({sparse/total*100 if total > 0 else 0:.1f}%)")
        slurm_safe_print(f"Dense steps:      {dense:,} ({dense/total*100 if total > 0 else 0:.1f}%)")
        slurm_safe_print(f"{'='*60}\n")

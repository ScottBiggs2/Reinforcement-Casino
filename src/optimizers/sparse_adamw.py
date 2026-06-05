import torch
import time
from src.utils.mask_manager import SparseMaskManager
from src.kernels.indexed_sparse_adam import triton_indexed_sparse_adamw_step

class SparseAdamW(torch.optim.Optimizer):
    """
    Custom AdamW optimizer with Triton-accelerated sparse updates.

    OPTIMIZATIONS:
    1. Pre-initialized optimizer states (no lazy init overhead)
    2. Indexed sparse kernel using gather/scatter operations
    3. Only processes non-zero mask elements (~2.5% with 97.5% sparsity)
    4. Precomputed bias corrections to avoid kernel recompilation

    NOTE: Only the element-wise indexed kernel is used (no BSR).  BSR block
    approximation degrades to near-dense updates when element sparsity is
    spread across a matrix (e.g. random / oracle masks at 2.5% density have
    nearly every 16×16 block active), which corrupts sparse training semantics.
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
    ):
        self.param_to_name = {}
        params = []
        _seen_ptrs: set = set()
        _n_tied_skipped = 0
        for name, param in named_params:
            ptr = param.data_ptr()
            if ptr in _seen_ptrs:
                # Tied weight — skip duplicate to avoid double updates and
                # inflated gradient norm in the clip pass.
                _n_tied_skipped += 1
                continue
            _seen_ptrs.add(ptr)
            params.append(param)
            self.param_to_name[id(param)] = name
        if _n_tied_skipped:
            print(f"SparseAdamW: skipped {_n_tied_skipped} tied-weight duplicate(s) "
                  f"(tie_word_embeddings or shared storage).")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.mask_manager = mask_manager
        self.block_size = block_size
        self.mlp_only = mlp_only
        self.max_grad_norm = max_grad_norm

        self.stats = {
            'sparse_steps': 0,
            'dense_steps': 0,
            'nan_warnings': [],
        }

        # Pre-initialize all optimizer states upfront (always eager).
        print(f"\nPre-initializing optimizer states for all parameters...")
        init_start = time.time()
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        init_time = time.time() - init_start
        print(f"✓ Optimizer states pre-initialized in {init_time:.2f}s")
        print(f"✓ SparseAdamW optimizer ready")
        print(f"  MLP-only: {mlp_only}")
        print(f"  Block size: {block_size}")
        print(f"  Using indexed sparse kernels: TRUE")
        print(f"  Local Gradient Clipping enabled: max_norm={self.max_grad_norm}")

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step using Triton kernels."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Per-parameter gradient clipping (protects momentum from spikes).
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(p, self.max_grad_norm)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_name = self.param_to_name.get(id(p), None)

                if param_name is None:
                    self._dense_step(p, group)
                    continue

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
        """Sparse update using the element-wise indexed Triton kernel."""
        grad = param.grad
        mask = self.mask_manager.get_mask(param_name)
        nonzero_indices = self.mask_manager.get_nonzero_indices(param_name)

        if mask is None or nonzero_indices is None:
            print(f"WARNING: Missing mask or indices for {param_name}, falling back to dense")
            self._dense_step(param, group)
            return

        if mask.shape != param.shape:
            print(f"WARNING: Shape mismatch for {param_name} "
                  f"(param {param.shape} vs mask {mask.shape}), falling back to dense")
            self._dense_step(param, group)
            return

        state = self.state[param]
        state['step'] += 1

        try:
            triton_indexed_sparse_adamw_step(
                param=param,
                grad=grad,
                nonzero_indices=nonzero_indices,
                exp_avg=state['exp_avg'],
                exp_avg_sq=state['exp_avg_sq'],
                lr=group['lr'],
                beta1=group['betas'][0],
                beta2=group['betas'][1],
                eps=group['eps'],
                weight_decay=group['weight_decay'],
                step=state['step'],
                block_size=self.block_size,
            )
        except Exception as e:
            print(f"ERROR in indexed Triton kernel for {param_name}: {e}")
            print(f"  Falling back to dense update")
            self._dense_step(param, group)
            return

        self.stats['sparse_steps'] += 1

    def _dense_step(self, param, group):
        """Standard dense AdamW update (fallback for non-masked params)."""
        grad = param.grad
        state = self.state[param]
        state['step'] += 1

        beta1, beta2 = group['betas']

        param.mul_(1 - group['lr'] * group['weight_decay'])

        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

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
            print(f"\n{'='*60}")
            print("WARNING: NaN/Inf DETECTED IN PARAMETERS")
            print(f"{'='*60}")
            for name, nan_count, inf_count in nan_params:
                print(f"  {name}: NaN={nan_count}, Inf={inf_count}")
            print(f"{'='*60}\n")
            self.stats['nan_warnings'] = nan_params
        else:
            print("\n✓ No NaN/Inf detected in parameters\n")

    def print_stats(self):
        """Print optimizer statistics."""
        total = self.stats['sparse_steps'] + self.stats['dense_steps']
        sparse = self.stats['sparse_steps']
        dense = self.stats['dense_steps']

        print(f"\n{'='*60}")
        print(f"SPARSE ADAMW OPTIMIZER STATISTICS")
        print(f"{'='*60}")
        print(f"Total steps:  {total:,}")
        print(f"Sparse steps: {sparse:,} ({sparse/total*100 if total > 0 else 0:.1f}%)")
        print(f"Dense steps:  {dense:,} ({dense/total*100 if total > 0 else 0:.1f}%)")
        print(f"{'='*60}\n")

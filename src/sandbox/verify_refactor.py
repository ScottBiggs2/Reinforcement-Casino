
import sys
import os
import torch

# Add src to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

print(f"Debug: Added {root_path} to sys.path")
# print(f"Debug: sys.path: {sys.path}")
print("Verifying imports...")

try:
    from src.kernels.bsr_backward import sparse_weight_gradient_triton
    print("✓ src.kernels.bsr_backward imported")
except ImportError as e:
    print(f"✗ Failed to import src.kernels.bsr_backward: {e}")

try:
    from src.kernels.indexed_sparse_adam import triton_indexed_sparse_adamw_step
    print("✓ src.kernels.indexed_sparse_adam imported")
except ImportError as e:
    print(f"✗ Failed to import src.kernels.indexed_sparse_adam: {e}")

try:
    from src.optimizers.sparse_adamw import SparseAdamW
    print("✓ src.optimizers.sparse_adamw imported")
except ImportError as e:
    print(f"✗ Failed to import src.optimizers.sparse_adamw: {e}")

try:
    from src.mlps.bsr_sparse_mlp import SparseLinearLayer, replace_linear_modules
    print("✓ src.mlps.bsr_sparse_mlp imported")
except ImportError as e:
    print(f"✗ Failed to import src.mlps.bsr_sparse_mlp: {e}")

try:
    from src.utils.mask_manager import SparseMaskManager
    print("✓ src.utils.mask_manager imported")
except ImportError as e:
    print(f"✗ Failed to import src.utils.mask_manager: {e}")

try:
    from src.utils.data_utils import load_dpo_dataset, dpo_collator_fn
    print("✓ src.utils.data_utils imported")
except ImportError as e:
    print(f"✗ Failed to import src.utils.data_utils: {e}")

try:
    from src.utils.logging_utils import FlexibleCheckpointCallback, CSVLoggerCallback
    print("✓ src.utils.logging_utils imported")
except ImportError as e:
    print(f"✗ Failed to import src.utils.logging_utils: {e}")

print("\nVerifying Instantiation (Dummy)...")

try:
    layer = SparseLinearLayer(128, 128, block_size=16, mask=None)
    print("✓ SparseLinearLayer instantiated")
except Exception as e:
    print(f"✗ SparseLinearLayer instantiation failed: {e}")

print("\nVerification Complete.")

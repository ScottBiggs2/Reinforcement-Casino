# Complexity Analysis: Sparse vs. Dense Training

To understand the true performance characteristics of our pipelines, we must model two distinct GPU bottlenecks:
1. **Compute Complexity (FLOPs)**: How many multiplication-additions the Tensor Cores perform. This dominates the **Backward Pass** where large matrices are multiplied.
2. **Memory Complexity (Bytes Moved)**: How much data is read from or written to VRAM. This dominates the **Optimizer Step**, which performs very simple math on massive arrays.

### Definitions
*   $B$: Batch size $\times$ Sequence length (total tokens).
*   $D_{in}$: Input dimension of the linear layer.
*   $D_{out}$: Output dimension of the linear layer.
*   $P$: Total parameters in the layer ($D_{in} \times D_{out}$).
*   $S$: Sparsity fraction (e.g., $S = 0.975$ means 97.5% zeros).
*   $A$: Active fraction ($1 - S$, e.g., $0.025$).

---

## Part 1: Backward Pass Computations (Gradient of Weights)
The equation is: `grad_W = grad_output.T @ input`
*   `grad_output`: Shape $(B, D_{out})$
*   `input`: Shape $(B, D_{in})$
*   `grad_W`: Shape $(D_{out}, D_{in})$

### 1a. Dense Backprop (PyTorch Native)
PyTorch performs a full, unmasked matrix multiplication using `cuBLAS`.
*   **Compute (FLOPs)**: $O(B \cdot D_{in} \cdot D_{out})$ math operations.
*   **Memory (Reads)**: Requires sliding over both $B \times D_{out}$ and $B \times D_{in}$ matrices, caching blocks in SRAM. Heavy, contiguous reads.
*   **Bottleneck**: Strictly Compute-Bound.

### 1b. Sparse BSR Backprop (Our Triton Kernel)
Our kernel checks if a block of output is entirely zeroed by the mask. If so, it instantly skips the matrix multiplication loop for that block.
*   **Compute (FLOPs)**: $O(A \cdot B \cdot D_{in} \cdot D_{out})$.
    *   *Example*: At 97.5% sparsity ($A = 0.025$), we execute **2.5%** of the mathematical operations compared to PyTorch dense.
*   **Memory (Reads)**: We only load the columns of `grad_output` and rows of `input` required to compute the active blocks.
*   **Bottleneck**: Can shift from Compute-Bound to Memory-Bound. If $A$ is extremely small, the overhead of querying the mask and launching threads dominates the actual dot-products.

---

## Part 2: Optimizer Step Computations
The optimizer applies updates element-wise. Element-wise operations are notoriously **Memory-Bandwidth Bound** because the math (e.g., $x + y$) takes microscopic fractions of a nanosecond, while fetching $x$ and $y$ from GDDR6 VRAM is comparatively very slow. 

For these calculations, assume all parameters/gradients are $32$-bit floats (4 bytes per element). Therefore, a fully dense read of parameters moves $4P$ bytes.

### 2a. Dense SGD
Math: `param = param - lr * grad`
*   **Memory Accessed per element**:
    1. Read `param`
    2. Read `grad`
    3. Write new `param`
*   **Total Memory Movement**: $3 \times 4P = 12P$ bytes.
*   **Time Complexity**: $O(P)$.

### 2b. Dense AdamW
Math involves momentum (`exp_avg`) and variance (`exp_avg_sq`).
*   **Memory Accessed per element**:
    1. Read `param`
    2. Read `grad`
    3. Read `exp_avg` (m)
    4. Read `exp_avg_sq` (v)
    5. Write new `exp_avg`
    6. Write new `exp_avg_sq`
    7. Write new `param`
*   **Total Memory Movement**: $7 \times 4P = 28P$ bytes.
*   **Time Complexity**: $O(P)$. At 28P memory movement, Dense AdamW is drastically slower than SGD purely due to pulling moving averages over the memory bus.

### 2c. Sparse AdamW (Indexed Gather/Scatter)
Instead of streaming $P$ elements continuously, the Triton kernel takes a list of $A \cdot P$ non-zero indices, and jumps directly to those memory addresses to do the reading and writing.
*   **Memory Accessed per element (ONLY for active indices)**: Same 7 steps as Dense AdamW.
*   **Total Memory Movement**: $7 \times 4(A \cdot P) = 28 A P$ bytes.
*   *Example*: At 97.5% sparsity ($A = 0.025$), Memory movement drops from $28P$ to **$0.7P$** bytes!
*   **Time Complexity**: $O(A \cdot P)$.
*   **Caveat**: Random memory access (Gather/Scatter) is less efficient than contiguous streaming (coalesced reads). A GPU might fetch 128 bytes of surrounding memory just to get the 4 bytes we requested at our index. Thus, the effective memory bandwidth used is worse than $0.7P$ in practice, but still astoundingly less than $28P$.

---

## Summary Matrix: The Combinations

Here is estimated the theoretical cost of updating the network weights per batch.

| Combination | Backprop FLOPs | Optimizer Memory I/O | Practical Verdict |
| :--- | :--- | :--- | :--- |
| **Dense Bwd + Dense AdamW** | $1.0 \times$ Math | $28P$ bytes | Baseline. Extremely compute-heavy and memory-heavy. |
| **Dense Bwd + SGD** | $1.0 \times$ Math | $12P$ bytes | Fast optimizer step, but backward pass is still huge compute. |
| **Dense Bwd + Sparse AdamW** | $1.0 \times$ Math | $28 A P$ bytes | Computes gigabytes of zeros in the backward pass math, then throws them away in the optimizer. Very inefficient backprop, lightning-fast optimizer. |
| **BSR Bwd + Dense AdamW** | $A \times$ Math ($\sim 2.5\%$) | $28P$ bytes | Fast backprop, but AdamW loads gigabytes of zeros, calculates zeros, and writes zeros back to VRAM. (Also destroys pre-trained weights due to weight-decay on frozen params). |
| **BSR Bwd + Sparse AdamW** | $A \times$ Math ($\sim 2.5\%$) | $28 A P$ bytes | **The Holy Grail**. Only multiplies non-zero mask matrices (bypassing massive FLOPs), then only gathers/scatters active indices (bypassing massive gigabytes of memory transfer). |

## Why Sparse Bwd + Sparse AdamW is Critical for Scale
When scaling up models (e.g., Llama-3 8B), $D_{in}$ and $D_{out}$ become thousands.
- $P$ becomes astronomically large.
- $B \cdot D_{in} \cdot D_{out}$ backward pass multiplications become the primary bottleneck of training step times.
- If $A$ is 0.025, BSR Backprop effectively reduces the GPU mathematics by an entire order of magnitude (97.5%), and Sparse AdamW rescues the VRAM Bandwidth by 97.5%. 
- Ultimately, this is the only combination that realizes true sub-linear scaling during fine-tuning.

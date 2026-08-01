# Cold Start Subnetwork Discovery: Research & Plan

## Objective
Identify a sparse subnetwork (mask) for the model **without training steps**, using only a small set of "warm-up" or "calibration" samples. The goal is to find the "Winning Ticket" or task-specific subnetwork via inference-time analysis (activations, gradients, or causal tracing).

## Constraints & Requirements
- **Inputs**: Pretrained Model, Small Dataset (few samples from DPO dataset).
- **Process**: Inference-only (forward passes, hook-based analysis, potentially gradient *calculation* w.r.t inputs/activations, but NO weight updates).
- **Output**: Binary mask file (`.pt`) compatible with `SparseMaskManager` (same format as `even_better_mask_finder.py`).

---

## Proposed Approaches

### 1. Activation Magnitude (Baseline)
*The "Fire Together, Wire Together" heuristic.*
- **Concept**: Neurons with high activation magnitudes on the target data are likely important.
- **Method**: 
    1. Run inference on $N$ samples (e.g., 64).
    2. Record moving average of activation magnitudes $|A_l|$ for each layer $l$.
    3. **Weight Mapping**: An MLP weight $W_{in}$ connects input $x$ to neuron $n$. If neuron $n$ is active, its incoming column $W_{:, n}$ and outgoing row $W_{n, :}$ are preserved.
    4. **Pros**: Extremely fast, zero gradients required.
    5. **Cons**: Ignores task relevance (might just pick "always active" polysemantic neurons).

### 2. Concept Activation Vectors (CAV) - *User Requested*
*Identify directions in activation space that correlate with the task.*
- **Concept**: [Kim et al., 2018](https://arxiv.org/abs/1711.11279). Use a linear probe to separate "Task" activations (e.g., DPO chosen responses) from "Background" activations (e.g., generic text or rejected responses).
- **Method**:
    1. **Data**: Set $P$ (Positive/Chosen) and $N$ (Negative/Rejected or Random).
    2. **Collection**: Extract activations $A_l$ for all layers.
    3. **Probe**: Train linear classifier $f_l(a) = v_l^T a + b_l$ to distinguish $P$ vs $N$. Vector $v_l$ is the CAV.
    4. **Scoring**: 
        - **Neuron Importance**: The magnitude of the CAV coefficient corresponding to the neuron $|v_{l, i}|$. High weight in classifier $\implies$ neuron is critical for distinguishing task.
    5. **Weight Mapping**: Create mask based on neuron importance scores.
- **Pros**: Task-specific signal. Uses the DPO data structure (Chosen vs Rejected) naturally.
- **Cons**: Requires training probes (lightweight).

### 3. SNIP / GraSP (Inference-Time Saliency)
*Gradient-based sensitivity pruning.*
- **Concept**: [Lee et al., 2018](https://arxiv.org/abs/1810.02340). Compute importance of weights based on their sensitivity to the loss *at initialization* (before training).
- **Method**:
    1. Compute DPO Loss $\mathcal{L}_{DPO}$ for a batch of input data.
    2. Backward pass to compute gradients $\nabla_W \mathcal{L}$. (Do NOT update weights).
    3. **Score**: $S = |W \odot \nabla_W \mathcal{L}|$ (Connection Sensitivity).
    4. Mask weights with top-k scores.
- **Pros**: Theoretically grounded "Lookahead" to what training *would* change. Directly scores 2D weights (no heuristic neuron-to-weight mapping needed).
- **Cons**: Requires backward pass (memory intensive).

### 4. ROME (Rank-One Model Editing) / Causal Tracing
*Locate knowledge storage.*
- **Concept**: [Meng et al., 2022](https://arxiv.org/abs/2202.05262). ROME computes a rank-one update $uv^T$ to modify a specific fact.
- **Method**:
    1. Treat the "few samples" as facts we want to "write" (or reinforce).
    2. Compute the theoretical ROME update matrix $\Delta W$ for the MLP layers.
    3. **Score**: Use $|\Delta W|$ as the importance score.
- **Pros**: Specifically targets MLP layers (where knowledge is stored).
- **Cons**: Computationally expensive (inverse covariance matrices). ROME is usually single-fact; batching is complex (MEMIT).

---

## Recommended Plan: "Activation-Guided Subnetworks"

We will implement a modular tool `src/cold_start/inference_mask_finder.py` that supports multiple scoring strategies.

### Step 1: Data Collection & Hooks
Create a `FeatureExtractor` class that hooks into `SparseLinearLayer` (or standard Linear) to capture inputs/outputs during inference.

### Step 2: Implement "CAV-Based" Scoring (Priority)
Since we have DPO pairs (Chosen/Rejected), CAV is a perfect fit.
- **Positive Setup**: Activations from `Chosen` responses.
- **Negative Setup**: Activations from `Rejected` responses.
- **Score**: Train Linear SVM/Logistic on each layer. Weights of the classifier become neuron importance scores.
- **Expansion**: Broadcast neuron scores 1D $\to$ 2D weight masks.

### Step 3: Implement "SNIP" Scoring (Secondary)
As a robust fallback.
- Compute one backward pass of DPO loss.
- Score $= |grad| * |weight|$.

### Step 4: Output Generation
Ensure output uses `torch.save({"masks": masks, "metadata": ...})` compatible with `SparseMaskManager`.

## Directory Structure
```
src/cold_start/
├── __init__.py
├── research_plan.md
├── inference_mask_finder.py    # Unified entry: --method fisher/cav/snip/activation
└── utils/
    ├── activation_hooks.py     # Hook logic
    ├── cav_probes.py          # Linear probe training
    └── snip_scorer.py         # Gradient scorer
```

## References
- **CAV**: *Interpretability Beyond Feature Attribution* (Kim et al., 2018)
- **SNIP**: *SNIP: Single-shot Network Pruning based on Connection Sensitivity* (Lee et al., 2018)
- **ROME**: *Locating and Editing Factual Associations in GPT* (Meng et al., 2022)

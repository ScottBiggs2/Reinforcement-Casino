\section{Methods}
\label{methods}

We study one-shot procedures for identifying task-relevant subnetworks in large language models without iterative pruning. Let $\theta$ denote the full parameter vector and let $\mathcal{M} \in \{0,1\}^{|\theta|}$ be a binary mask of the same dimensionality, where $\mathcal{M}_i = 1$ denotes an active weight and $\mathcal{M}_i = 0$ denotes a pruned weight. The resulting subnetwork is $\theta_{\mathcal{M}} = \mathcal{M} \odot \theta$, where $\odot$ is the elementwise product. Our goal is to construct $\mathcal{M}$ at a target sparsity level $\rho$, defined as the fraction of weights excluded from the subnetwork. This setup follows the broader pruning and lottery-ticket literature while avoiding the cost of iterative search at LLM scale \citep{han2015learning,frankle2018the,burkholz2022on,mukherjee2025reinforcement, lee2019snip, Wang2020Picking}.

Our framework has two stages. First, a scoring rule assigns each parameter a saliency value, denoted $S = \mathcal{S}(\theta) \in \mathbb{R}^{|\theta|}$. Second, a mask-construction operator maps those scores to a binary mask, denoted $\mathbb{M}(S) = \mathcal{M}$, by retaining the highest-scoring weights under a prescribed sparsity budget. 

% \scott{removing compliment mask terminology, we don't need it}
% When needed, we also consider the complement mask $\widetilde{\mathcal{M}} = \mathbf{1} - \mathcal{M}$ to test whether performance is concentrated in the selected subnetwork or in the weights it excludes.

During preference optimization, we evaluate the dense model while restricting parameter updates to the active subnetwork $\theta_{\mathcal{M}}$. This isolates the effect of the selected weights while keeping the forward computation, data, and objective fixed across sparse and dense conditions. 

\subsection{Scoring}
Iterative pruning is often prohibitively expensive at the scale of contemporary language models, so we focus on few-shot scoring rules that can be computed either from a short dense training trajectory or from a small calibration set \citep{frankle2018the,mukherjee2025reinforcement, lee2019snip, Wang2020Picking}. As \cite{mukherjee2025reinforcement} shows empirically, these subnetworks solidify by the weight-update magnitude measure early in training. 

% \subsubsection{Magnitude}
% Warm-start magnitude scoring measures how much each parameter moves along a dense optimization trajectory. Given checkpointed updates $\Delta \theta_t$, the score for weight $i$ is the accumulated absolute displacement,
% \[
%     s_i = \sum_t |(\Delta \theta_t)_i|.
% \]
% Weights with larger cumulative movement are treated as more task-relevant. Detailed pseudocode for this and the other subnetwork scoring routines is deferred to Appendix~\ref{app:subnetwork-algorithms}.

% \subsubsection{Momentum}
% Momentum-based scoring emphasizes parameters whose updates are not only large but also directionally consistent over time. It aggregates short-horizon update velocities $v_{i,t} = (\Delta \theta_t)_i - (\Delta \theta_{t-1})_i$ and assigns larger scores to weights that move persistently in a stable direction. The score is computed as:
% \[
%     s_i = |\mu_i| \left(1 + \frac{|\mu_i|}{\sigma_i + \epsilon}\right),
% \]
% where $\mu_i$ and $\sigma_i$ are the mean and standard deviation of the velocities over a sliding window. This criteria favors parameters that exhibit "meaningful" directional progress while penalizing those with high-variance or oscillatory behavior.

% \subsubsection{Fisher on Weight Deltas}
% Fisher-on-deltas combines average displacement with empirical variance along the training trajectory. This approach treats the optimization trajectory as a sequence of observations and utilizes the second moment of the updates as a proxy for parameter importance:
% \[
%     s_i = \text{Var}[(\Delta \theta)_i] + |\mathbb{E}[(\Delta \theta)_i]|.
% \]
% Intuitively, it favors weights that both move substantially and exhibit sustained sensitivity across the optimization steps.

% \subsubsection{Empirical Fisher on Prompts}
% Cold-start empirical Fisher scoring dispenses with a dense warm-start trajectory and instead estimates parameter saliency directly from gradient statistics on a calibration set. Concretely, it uses squared gradients under the training objective as a diagonal proxy for parameter sensitivity, following the classical second-order pruning literature \citep{lecun1990optimal}.

% \subsubsection{CAV/SNIP on Activations}
% We also consider two activation- or gradient-based one-shot criteria. 

% \paragraph{Concept Activation Vectors (CAV).} We identify directions in hidden-state space that distinguish task-relevant positive and negative examples. Specifically, for each MLP layer, we train a logistic regression probe on the activations $\mathbf{h} \in A^{+}$ and $\mathbf{h} \in A^{-}$ to obtain a concept vector $\beta$. The parameter scores are derived from the magnitude of the probe coefficients, reflecting each neuron's contribution to the high-level concept:
% \[
%     s_j = \frac{|\beta_j|}{\max |\beta|} + \lambda \frac{\mathbb{E}[|h_j|]}{\max \mathbb{E}[|h|]},
% \]
% where $\lambda$ balances the concept sensitivity with raw activation magnitude \citep{pmlr-v80-kim18d}.

% \paragraph{SNIP.} Single-shot Network Pruning (SNIP) measures connection saliency through the first-order effect of removing a weight on the preference loss $\mathcal{L}$. It is computed using a single backward pass on a calibration set:
% \[
%     s_i = |w_i \odot \nabla_{w_i} \mathcal{L}|.
% \]
% This criteria identifies weights that are most "sensitive" to the optimization objective without requiring any training steps \citep{lee2019snip}.

\subsubsection{Random Masks}

The random mask is constructed with random sampling from $\mathcal{U}(0, n)$ with probability $\rho$. This can be done at the element/weight level, or at the block level. A uniform distribution in weight space is a strong prior due to the permutation invariance of transformer weights disallowing persistent local or global structure under functionally equivalent permutations of weights. \scott{citations needed}. 

\subsubsection{Oracle Masks}

The oracle, or ground truth, mask is constructed with magnitude scoring over a complete dense training run. This measures how much each parameter moves along a dense optimization trajectory. Given checkpointed updates $\Delta \theta = |\theta_F - \theta_0|$, the score for weight $i$of $n$ is the absolute displacement,
\[
    s_i = |\Delta \theta_i|
\]
Weights with larger cumulative movement are treated as more task-relevant. Detailed pseudocode for this and the other subnetwork scoring routines is deferred to Appendix~\ref{app:subnetwork-algorithms} \scott{check and update appendix}.

\subsubsection{SNR Reweighted GraSP}

Gradient Signal Preservation (GraSP) scores parameters by their contribution to preserving gradient flow through a pruned network. We adapt GraSP to our preference-optimization setting by (i) scoring with objectives aligned to the downstream training loss and (ii) optionally reweighting scores by an empirical gradient signal-to-noise ratio (SNR) computed over calibration minibatches.

\paragraph{Base GraSP score.}
Let $W_\ell \in \mathbb{R}^{m\times n}$ denote a scored weight matrix (we score 2D weight tensors). Given a calibration set, we first estimate the mean gradient $g_{\mathrm{avg}} = \mathbb{E}_b[g^{(b)}]$ where $g^{(b)} = \nabla_W \mathcal{L}^{(b)}$ is the minibatch gradient under an objective $\mathcal{L}$. We then estimate a Hessian--gradient product by computing, for each minibatch, an HVP $H^{(b)} g_{\mathrm{avg}}$ via automatic differentiation and averaging across calibration minibatches. The resulting GraSP saliency score is computed elementwise as
\[
S(W) \;=\; -\bigl(H g_{\mathrm{avg}}\bigr)\odot W.
\]
We use objectives that match the downstream setting: (i) a language-modeling objective based on causal cross-entropy on calibration sequences, and (ii) a preference objective based on a DPO-style chosen-vs-rejected margin loss.

\paragraph{SNR reweighting of scores.}
To reduce sensitivity to noisy minibatch gradients, we optionally compute an SNR-based multiplier from calibration minibatch gradients and apply it multiplicatively to the base score. In \textit{per-tensor} mode, for each weight matrix we track scalar moments of the minibatch gradient magnitude, $\mu=\mathbb{E}_b[\|g^{(b)}\|_1/(mn)]$ and $\sigma=\mathrm{Std}_b[\|g^{(b)}\|_1/(mn)]$, and define $\mathrm{snr}=|\mu|/\sigma$. In \textit{per-weight} mode, we track elementwise moments and define $\mathrm{snr}_{ij}=|\mu_{ij}|/(\sigma_{ij}+\epsilon)$. We transform the SNR into a nonnegative multiplier $m=f(\mathrm{snr})$ using a simple monotone transform (identity, $\log(1+\mathrm{snr})$, or clamping), and reweight scores as
\[
S_{\mathrm{snr}}(W) \;=\; S(W)\odot m,
\]
where $m$ is either a scalar (per-tensor) or a tensor of the same shape as $W$ (per-weight). Per-weight SNR accumulation is substantially more memory intensive than per-tensor accumulation, so we treat it as an optional setting rather than a default.




\subsection{Mask Construction}
\label{mask-construction}
After obtaining scores $S = \mathcal{S}(\theta)$, we construct a binary mask $\mathcal{M} = \mathbb{M}(S)$ by retaining the top-scoring weights under a target sparsity level $\rho$. We analyze two primary pooling strategies:

\paragraph{Global Pooling.} All eligible weights across the target modules are ranked jointly. This allows the model to allocate its parameter budget non-uniformly, concentrating capacity where scores indicate the greatest sensitivity. To reduce pathological layer collapse at extreme sparsities, we use a \textbf{hybrid-global} operator: each scored tensor retains a small \emph{keep floor} (a minimum fraction of its elements), and the remaining keep budget is then allocated by a global top-$k$ selection across the full set of scored parameters.

\paragraph{Local Pooling.} The sparsity budget is applied independently to each parameter tensor. This ensures that every layer retains exactly $(1-\rho)$ of its weights, providing a uniform distribution of capacity. While less flexible than global pooling, local selection can be more stable and easier to implement for hardware-structured sparsity.

\paragraph{Exact selection at scale.}
For very large models, constructing a monolithic GPU score vector and applying a naive GPU top-$k$ can exceed memory limits or hit implementation constraints. In these regimes, we perform \textbf{exact chunked selection} on host memory: scores are processed in chunks, a narrow threshold interval is refined via histogram passes, and the final top-$k$ boundary is resolved without materializing multiple full copies of the model-wide score vector.

\paragraph{Element vs.\ block masks.}
We consider two mask granularities. In \textbf{element} masks, selection is applied directly to weight-shaped score tensors. In \textbf{block} masks, we first pool elementwise scores into a block grid of $B\times B$ tiles using either a mean or max reduction, apply top-$k$ selection on the block grid, and then expand selected blocks back to the full weight shape. Since the sparsity budget is enforced at the block-grid level, the realized elementwise sparsity after expansion and edge cropping can differ slightly from the nominal target.

\subsection{Mask Comparisons}
Comparing subnetworks across tasks, scoring rules, and sparsity levels is part of our methodology. We therefore report overlap and representation-similarity statistics at the layer-wise, component-wise, and aggregate levels \citep{pmlr-v97-kornblith19a}.

\subsubsection{Jaccard Score}
For binary masks $\mathcal{M}_1$ and $\mathcal{M}_2$, we measure overlap using the Jaccard score,
\begin{equation}
    J = \frac{\mathcal{M}_1 \cap \mathcal{M}_2}{\mathcal{M}_1 \cup \mathcal{M}_2}.
\end{equation}
We report this quantity in aggregate and after decomposition by layer and component type. The corresponding random-mask baseline is given in Appendix~\ref{app:mask-comparison-details}.

\subsubsection{Centered Kernel Alignment}
To compare the representations induced by different subnetworks, we also use linear centered kernel alignment (CKA), which is invariant to isotropic rescaling and orthogonal transformation of the representation basis \citep{pmlr-v97-kornblith19a}. We apply CKA both layer-wise and in aggregate to quantify representational similarity beyond binary overlap.

\subsubsection{Linear Probe Interpretability}
To detect human-interpretable differences in how subnetworks reshape internal model logic, we employ a linear probing framework across the MLP activations of the LLM \citep{alain2016understanding,tenney2019bert}. For a given layer $l$, we extract the activation vector $\mathbf{h}_l$ and train a logistic regression classifier to predict specific task-relevant properties. Our analysis focuses on four distinct categories: \scott{cite these datasets}
\begin{enumerate}
    \item \textbf{Syntax}: Predicting grammatical categories and subject-verb agreement consistency.
    \item \textbf{Semantics}: Identifying sentiment polarity and topical coherence.
    \item \textbf{Factual Knowledge}: Distinguishing between entity types (e.g., scientists vs. locations) and verifying factual relations.
    \item \textbf{Mathematics}: Assessing the correctness of arithmetic operations and identifying reasoning errors.
\end{enumerate}
We collect activations from 60 balanced examples (30 positive, 30 negative) per property. By comparing probe accuracies on the base model versus models trained on different sparse subnetworks, we can quantify which functional capabilities are preserved, enhanced, or degraded by the selection criteria. This allows us to move beyond aggregate performance metrics and understand the "functional signature" of each lottery ticket.

\subsection{Sparse Backpropagation}
Our sparse-training experiments keep the forward computation fixed while restricting gradient updates to the active subnetwork $\theta_{\mathcal{M}}$. This design allows us to compare dense and sparse optimization under a common objective and data distribution, isolating the effect of which parameters are allowed to adapt.

To achieve hardware efficiency during sparse fine-tuning, we implement custom Triton kernels that exploit block structure in the update mask. Concretely, we use a Block Sparse Row (BSR) representation with a block size $B\times B$ (typically $16\times 16$) for the weights being updated. We keep the \textbf{forward pass dense} (standard linear layers) for stability and simplicity, while accelerating the \textbf{backward pass} and optimizer update by avoiding computation on masked-out blocks.

\paragraph{Block-sparse weight gradients.}
For a linear layer with input $\mathbf{X}$ and output gradient $\boldsymbol{\delta}$, the weight gradient is $\nabla_W \mathcal{L}=\boldsymbol{\delta}^\top \mathbf{X}$. Under a BSR mask, we compute this outer-product accumulation only for the active $B\times B$ blocks of $W$, skipping blocks that are entirely masked. At high sparsities this avoids most of the memory traffic and multiply--accumulate work in the weight-gradient computation. For extreme sparsity (few active blocks), we further shard work over the batch dimension to maintain GPU occupancy.

\paragraph{Dense vs.\ sparse input gradients.}
The gradient with respect to the layer input can be computed densely as $\nabla_{\mathbf{X}}\mathcal{L}=\boldsymbol{\delta}W$. When the updated weights are block-masked, we optionally compute $\nabla_{\mathbf{X}}\mathcal{L}$ using a block-sparse kernel that multiplies only the active blocks; alternatively, we use the dense matmul path. This separates a correctness-preserving dense baseline from a fully sparse backward path and enables controlled ablations of where sparsity is introduced in backpropagation.

\paragraph{Sparse optimizer updates.}
Optimizer steps are typically memory-bandwidth bound, so updating dense moment buffers for parameters that are frozen by $\mathcal{M}$ wastes substantial memory traffic. We therefore use an \textit{indexed sparse optimizer} (Sparse AdamW) that only updates optimizer state and parameters at active locations. In the \textbf{element} regime, this is implemented via gather--update--scatter at a list of active parameter indices. In the \textbf{block} regime, we update only the active $B\times B$ blocks using a block-aware kernel that is coalesced and tensor-core friendly. For stability, we also apply gradient clipping using norms computed only over the active indices/blocks, so the clipping computation scales with the active fraction rather than the dense parameter count.
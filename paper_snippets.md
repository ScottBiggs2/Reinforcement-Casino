%% Pseudocode below uses \usepackage{algorithm} and \usepackage{algpseudocode} (or algorithmicx).

\section{Methods}
\label{methods}

We consider several methods for finding task oriented subnetworks within LLMs without iterative pruning. Instead, we proceed with the goal of constructing a binary mask $\mathcal{M}\in\mathbb{R}^\theta$, where $m_i \in \mathcal{M} = 1$ indicating active weights and $m_i \in \mathcal{M} = 0$ indicating weights outside the subnetwork. This mask is constructed in a two step process: 

1.) Scoring: Each weight is scored by a scoring function, $\mathcal{S}(\theta)=S\in \mathbb{R}^\theta$, where each $s_i\in S$ is the score associated with weight $\theta_i$. 

2.) These scores are fed into a mask construction routine (one-shot global importance ranking over scores; see \S\ref{mask-construction}) to construct $\mathcal{M}$ with a target sparsity level $\rho$, where $\rho$ is the proportion of weights \emph{excluded} from $\mathcal{M}$ (set to zero). Classical magnitude-based pruning \citep{han2015learning} and the contrast with iterative lottery-ticket style search \citep{frankle2018lottery} motivate this non-iterative family of methods at LLM scale.

To thoroughly and fairly investigate the responsible subnetwork in experiments, forward passes are done in the dense $f_\theta$ while masked optimization is applied to $f_{\mathcal{M} \odot \theta}$: we use Hugging Face TRL's \texttt{DPOTrainer} / \texttt{DPOConfig} for the Direct Preference Optimization objective \citep{rafailov2023dpo}, \texttt{SparseMaskManager} (\path{src/utils/mask_manager.py}) to load binary masks and precompute nonzero indices for gather/scatter-style updates, and Triton-accelerated \texttt{SparseAdamW} (\path{src/optimizers/sparse_adamw.py}) in the primary sparse training script \path{src/full_training/sparse_dpo_efficiency.py}. An alternate path injects block-sparse MLP layers with a custom backward (\path{src/full_training/sparse_dpo_bsr.py}). Standard dense AdamW decays all parameters; with a fixed mask, that behavior can incorrectly decay frozen weights, so masked runs use sparse or masked optimizers as appropriate (see project \path{readme.md}).

Additionally, the complement mask $\tilde{\mathcal{M}}$ is simply the mask $\mathcal{M}$ where each $\tilde{m}_i = !m_i$ , for each, $\tilde{m}_i \in \tilde{\mathcal{M}}$ and $m_i\in\mathcal{M}$. Alternatively, you can construct $\tilde{\mathcal{M}}$ with $\mathcal{M} \oplus \tilde{\mathcal{M}} = \mathbf{1}\in \mathbb{R}^\theta$. The complement mask has sparsity level $\tilde{\rho} = 1-\rho$. 


\subsection{Scoring}
Iterative pruning methods, typical in Lottery Ticket identification, are extremely expensive at LLM scale. Instead, we examine several alternative methods:

\subsubsection{Magnitude}
Warm-start magnitude scoring aggregates per-weight movement along a dense DPO trajectory. Checkpointed weight deltas $\Delta \theta_t$ (from \path{delta_logs_*}) are streamed without loading the full trajectory into memory; the score for each weight is the sum of absolute deltas, $s_i = \sum_t |(\Delta \theta_t)_i|$, implemented in \path{src/warm_start/even_better_mask_finder.py} as \texttt{compute\_absolute\_magnitude\_mask\_streaming}. Optional restrictions keep only MLP parameters (\texttt{--mlp\_only}).

\begin{algorithm}[t]
\caption{Warm-start magnitude scoring over streamed checkpoints}
\label{alg:warm-magnitude}
\begin{algorithmic}[1]
\Require Ordered checkpoint paths $\{\Delta \theta_1,\ldots,\Delta \theta_T\}$; optional filter on parameter names (e.g.\ MLP-only)
\Ensure Score tensor $S$ aligned with each parameter in $\theta$
\State Initialize $A \gets 0$ for every scored parameter shape
\For{each checkpoint $t = 1,\ldots,T$}
    \State Load $\Delta \theta_t$; optionally drop non-MLP tensors
    \For{each parameter tensor $p$}
        \State $A[p] \gets A[p] + |\Delta \theta_t[p]|$ \Comment{elementwise}
    \EndFor
\EndFor
\State $S \gets A$
\end{algorithmic}
\end{algorithm}

\subsubsection{Momentum}
Momentum-based scores use discrete velocities between consecutive delta checkpoints, $v_t = \Delta_t - \Delta_{t-1}$, kept in a sliding window of width $w$. Per weight, the implementation combines mean velocity magnitude with a consistency term (mean divided by standard deviation across the window) into a single nonnegative score; see \texttt{compute\_momentum\_mask\_streaming} in \path{src/warm_start/even_better_mask_finder.py}. If fewer than two steps are available, it falls back to magnitude scoring.

\begin{algorithm}[t]
\caption{Momentum scoring from delta velocities (sliding window)}
\label{alg:warm-momentum}
\begin{algorithmic}[1]
\Require Checkpoints $\{\Delta \theta_1,\ldots,\Delta \theta_T\}$; window size $w \ge 1$
\Ensure Score tensor $S$ per parameter
\If{$T < 2$}
    \State \Return magnitude scores (Alg.~\ref{alg:warm-magnitude})
\EndIf
\State Initialize empty deques $\mathcal{V}_p$ of length at most $w$ for each parameter $p$
\State $\Delta_{\mathrm{prev}} \gets \Delta \theta_1$
\For{$t = 2,\ldots,T$}
    \State $\Delta \gets \Delta \theta_t$
    \For{each parameter $p$}
        \State $v \gets \Delta[p] - \Delta_{\mathrm{prev}}[p]$ \Comment{velocity}
        \State Append $v$ to $\mathcal{V}_p$; pop front if $|\mathcal{V}_p| > w$
    \EndFor
    \State $\Delta_{\mathrm{prev}} \gets \Delta$
\EndFor
\For{each parameter $p$}
    \If{$|\mathcal{V}_p| < 2$}
        \State $S[p] \gets |\Delta_{\mathrm{prev}}[p]|$ \Comment{fallback}
    \Else
        \State Stack velocities into $V \in \mathbb{R}^{|\mathcal{V}_p| \times \cdots}$ along step
        \State $\mu \gets \mathrm{mean}(V)$, \quad $\sigma \gets \mathrm{std}(V) + \epsilon$
        \State $S[p] \gets |\mu| \odot \bigl(\mathbf{1} + |\mu|/\sigma \bigr)$ \Comment{elementwise}
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}

\subsubsection{Fisher on Deltas}
Fisher-on-deltas scoring treats the sequence of weight deltas across logged steps as a sample path and assigns each weight a score related to variability and mean displacement: approximately $\mathrm{Var}[\Delta] + |\mathbb{E}[\Delta]|$ per element, accumulated in streaming fashion (sums and sums of squares) in \texttt{compute\_fisher\_mask\_streaming} in \path{src/warm_start/even_better_mask_finder.py}.

\begin{algorithm}[t]
\caption{Fisher-on-deltas: variance plus mean magnitude per weight}
\label{alg:warm-fisher-delta}
\begin{algorithmic}[1]
\Require Checkpoints $\{\Delta \theta_1,\ldots,\Delta \theta_T\}$
\Ensure Score tensor $S$ per parameter
\State $c \gets 0$; initialize $S_\Sigma \gets 0$, $S_{\Sigma^2} \gets 0$ for each parameter (same shape as $\Delta \theta_t$)
\For{$t = 1,\ldots,T$}
    \State Load $\Delta \theta_t$; $c \gets c + 1$
    \For{each parameter $p$}
        \State $S_\Sigma[p] \gets S_\Sigma[p] + \Delta \theta_t[p]$
        \State $S_{\Sigma^2}[p] \gets S_{\Sigma^2}[p] + (\Delta \theta_t[p])^2$
    \EndFor
\EndFor
\For{each parameter $p$}
    \State $\mu \gets S_\Sigma[p] / c$, \quad $\mathrm{Var} \gets \max\bigl(0,\, S_{\Sigma^2}[p]/c - \mu^2\bigr)$
    \State $S[p] \gets \mathrm{Var} + |\mu|$ \Comment{elementwise}
\EndFor
\end{algorithmic}
\end{algorithm}

\subsubsection{Empirical Fisher (cold start)}
Cold-start empirical Fisher scores use diagonal sensitivity estimates from gradients under the preference objective, not activations alone. \path{src/cold_start/cold_mask_finder.py} computes squared-gradient accumulates (empirical Fisher) over DPO-style batches, with optional per-layer z-score normalization before masking. This is distinct from activation-based probes below; the subsection title avoids ``Fisher on activations'' to match the implementation. Second-order pruning lineage traces to classical sensitivity criteria \citep{lecun1990optimal}.

\begin{algorithm}[t]
\caption{Cold-start empirical Fisher (squared-gradient accumulation)}
\label{alg:cold-fisher}
\begin{algorithmic}[1]
\Require Model $f_\theta$; calibration batches $\mathcal{B}_1,\ldots,\mathcal{B}_K$ (e.g.\ tokenized chosen sequences); optional MLP-only filter
\Ensure Score tensor $F$ per parameter
\State Initialize $F[p] \gets 0$ for each trainable parameter $p$ to score
\For{$k = 1,\ldots,K$}
    \State $\mathcal{L} \gets \frac{1}{|\mathcal{B}_k|} \sum_{x \in \mathcal{B}_k} \ell(x;\theta)$ \Comment{language-modeling loss on batch}
    \State $\nabla_\theta \mathcal{L}$ via one backward pass
    \For{each scored parameter $p$}
        \State $F[p] \gets F[p] + \bigl(\nabla_p \mathcal{L}\bigr)^2$ \Comment{elementwise}
    \EndFor
\EndFor
\For{each $p$}
    \State $F[p] \gets F[p] / K$
\EndFor
\If{per-layer normalization}
    \For{each tensor $F[p]$}
        \State $F[p] \gets (F[p] - \mathrm{mean}(F[p])) / (\mathrm{std}(F[p]) + \epsilon)$ \Comment{or $0$ if $\mathrm{std}$ negligible}
    \EndFor
\EndIf
\end{algorithmic}
\end{algorithm}

\subsubsection{CAV/SNIP on Activations}
\textbf{CAV.} Concept Activation Vectors \citep{kim2018tcav} separate activations from task-positive vs.\ task-negative inputs; our cold-start pipeline (\path{src/cold_start/inference_mask_finder.py}, \path{src/cold_start/cav_cold_mask_finder.py}, \path{src/cold_start/utils/cav_probes.py}) uses chosen vs.\ rejected pairs from a small DPO calibration set to train linear probes and derive neuron- then weight-level importance.

\textbf{SNIP.} Single-shot connection sensitivity \citep{lee2019snip} scores weights by $|w \odot \nabla_w \mathcal{L}|$ after one backward pass without an optimizer step. Our \path{src/cold_start/utils/snip_scorer.py} applies this saliency to MLP weight matrices using a language-modeling loss on chosen sequences (see \texttt{SNIPScorer}).

\begin{algorithm}[t]
\caption{CAV-style neuron scores from chosen vs.\ rejected activations}
\label{alg:cav}
\begin{algorithmic}[1]
\Require Activations per layer $\ell$: matrices $A^{+}_\ell \in \mathbb{R}^{N_+ \times d_\ell}$, $A^{-}_\ell \in \mathbb{R}^{N_- \times d_\ell}$; blend weight $\lambda \ge 0$
\Ensure Per-layer neuron score vectors $s_\ell \in \mathbb{R}^{d_\ell}$ (later broadcast to MLP weight masks)
\For{each layer $\ell$}
    \State Form labels $y \gets (1,\ldots,1,0,\ldots,0)^\top$; rows $X \gets [A^{+}_\ell; A^{-}_\ell]$
    \State Standardize columns of $X$; fit L1-regularized logistic regression
    \State $\mathrm{cav} \gets |\beta| \in \mathbb{R}^{d_\ell}$ \Comment{classifier coefficients}
    \State $\mathrm{mag} \gets \mathrm{mean}_{\mathrm{rows}} |A^{+}_\ell|$ \Comment{per-neuron}
    \State $\tilde{\mathrm{cav}} \gets \mathrm{cav} / (\max \mathrm{cav} + \epsilon)$; $\tilde{\mathrm{mag}} \gets \mathrm{mag} / (\max \mathrm{mag} + \epsilon)$
    \State $s_\ell \gets \tilde{\mathrm{cav}} + \lambda \tilde{\mathrm{mag}}$
\EndFor
\State Map $s_\ell$ to binary masks on \texttt{gate\_proj}, \texttt{up\_proj}, \texttt{down\_proj} via \texttt{scores\_to\_masks} (global or per-layer top-$k$ on neurons)
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[t]
\caption{SNIP connection saliency (one backward, no optimizer step)}
\label{alg:snip}
\begin{algorithmic}[1]
\Require $f_\theta$; calibration texts; MLP weight set $\mathcal{W}$
\Ensure Saliency $S_w$ for each $w \in \mathcal{W}$
\State $\theta.\mathrm{grad} \gets 0$; \quad $\mathcal{L}_{\mathrm{tot}} \gets 0$; \quad $B \gets 0$
\For{each mini-batch of inputs}
    \State $\mathcal{L}_{\mathrm{tot}} \gets \mathcal{L}_{\mathrm{tot}} + \ell_{\mathrm{LM}}(\text{batch})$ \Comment{causal LM loss}
    \State $B \gets B + 1$
\EndFor
\State Backward once on $\mathcal{L}_{\mathrm{tot}} / B$ \Comment{mean loss over mini-batches}
\For{each MLP 2D weight $w \in \mathcal{W}$ with gradient}
    \State $S_w \gets |w \odot \nabla_w \mathcal{L}|$
\EndFor
\end{algorithmic}
\end{algorithm}

\subsection{Mask Construction}
\label{mask-construction}
After scores $S = \mathcal{S}(\theta)$ are produced, we map them to binary masks with $\mathbb{M}(S) = \mathcal{M}$ via \texttt{create\_mask\_from\_scores\_gpu\_efficient} in \path{src/utils/mask_utils.py}. The selector flattens scores across targeted parameters, chooses the top fraction of weights to \emph{keep} so that the fraction of zeros matches the target exclusion rate $\rho$, with three pooling modes: global ranking; global ranking with a small per-layer keep floor (\texttt{DEFAULT\_MIN\_LAYER\_KEEP\_RATIO}) to avoid layer collapse at extreme sparsity; or local per-tensor top-$k$ (\texttt{local\_pool}). Optional tie-break noise breaks score ties; for very large score vectors, CPU top-$k$ may be used to avoid GPU gather limits (see \texttt{\_CUDA\_TOPK\_SAFE\_NUMEL}). Metadata for the chosen mode is recorded via \texttt{pooling\_metadata} for reproducibility.

\begin{algorithm}[t]
\caption{Local per-tensor mask construction (uniform sparsity per weight matrix)}
\label{alg:mask-local}
\begin{algorithmic}[1]
\Require Score tensors $\{S_p\}$ for each parameter $p$; target exclusion fraction $\rho \in [0,1]$
\Ensure Binary masks $m_p \in \{0,1\}^{\mathrm{shape}(p)}$
\State $\alpha \gets 1 - \rho$ \Comment{fraction to keep}
\For{each tensor $p$}
    \State $k \gets \max(1,\lfloor \alpha \cdot \mathrm{numel}(S_p) \rfloor)$
    \State Optionally add tie-break noise to flattened $S_p$
    \State Let $I$ be indices of top-$k$ values of $\mathrm{vec}(S_p)$
    \State $m_p \gets 0$; set $m_p[i]=1$ for $i \in I$
\EndFor
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[t]
\caption{Global mask with optional per-layer keep floor, then global top-$k$}
\label{alg:mask-global-floor}
\begin{algorithmic}[1]
\Require Valid score tensors $\{S_p\}$; target exclusion $\rho$; floor ratio $r \ge 0$ (may be $0$); optional tie-break noise
\Ensure Binary masks $\{m_p\}$ with $\approx \rho$ fraction of zeros globally
\State $N \gets \sum_p \mathrm{numel}(S_p)$; \quad $k_{\mathrm{keep}} \gets \lfloor (1-\rho) N \rfloor$
\If{tie-break}
    \State Add small Gaussian noise to each $S_p$ (scale from score magnitudes)
\EndIf
\State Initialize all masks $m_p \gets 0$
\State \textbf{Floor pass:}
\For{each tensor $p$}
    \State $k_p \gets \lfloor r \cdot \mathrm{numel}(S_p) \rfloor$; if $\sum_q k_q > k_{\mathrm{keep}}$, scale all $k_q$ down proportionally
    \State $I_p^{\mathrm{floor}} \gets$ indices of top-$k_p$ values of $\mathrm{vec}(S_p)$
    \State Set $m_p[i]=1$ for $i \in I_p^{\mathrm{floor}}$
\EndFor
\State $k_{\mathrm{rem}} \gets k_{\mathrm{keep}} - \sum_p |I_p^{\mathrm{floor}}|$
\State \textbf{Global pass:} Select top-$k_{\mathrm{rem}}$ scores among remaining positions (excluding floor slots) via threshold search / histogram refinement on flattened scores, or single $\mathrm{topk}$ when $N$ is small enough
\State \Comment{Very large $N$ uses chunked histogram refinement and optional CPU $\mathrm{topk}$; see \texttt{\_create\_mask\_global\_chunked}}
\end{algorithmic}
\end{algorithm}


\subsection{Sparse Backpropagation}
The main efficiency experiments use \path{src/full_training/sparse_dpo_efficiency.py}: dense transformer forwards with masks applied so updates touch only masked weights, and Triton-accelerated kernels for indexed sparse AdamW \citep{triton}. Optimizer ablations include SGD, dense AdamW, and \texttt{sparse\_adamw}. A separate code path, \path{src/full_training/sparse_dpo_bsr.py}, replaces selected MLP layers with block-sparse (BSR) linear modules and custom autograd; project documentation notes that BSR backward correctness and performance are not yet fully validated. PyTorch's standard autograd supplies dense baselines \citep{paszke2019pytorch}.

%% Bibliography keys (replace with your .bib entries):
%% han2015learning: https://arxiv.org/abs/1506.02626
%% frankle2018lottery: https://arxiv.org/abs/1803.03635
%% rafailov2023dpo: https://arxiv.org/abs/2305.18290
%% lecun1990optimal: Optimal Brain Damage, NIPS 1990
%% kim2018tcav: http://arxiv.org/abs/1711.11279
%% lee2019snip: https://arxiv.org/abs/1810.02340
%% triton: Triton DSL / compiler (https://triton-lang.org/) — use @misc or vendor bib as you prefer
%% paszke2019pytorch: PyTorch, NeurIPS 2019 — https://arxiv.org/abs/1912.01703
%% TRL docs: https://huggingface.co/docs/trl

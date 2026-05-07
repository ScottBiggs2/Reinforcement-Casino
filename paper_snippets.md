\begin{algorithm}[h]
\caption{Default hybrid mask: global budget with per-tensor keep floor (\texttt{local\_pool}$=$ false, \texttt{min\_layer\_keep\_ratio}$=r$)}
\label{alg:mask-hybrid-default}
\begin{algorithmic}[1]
\Require Score tensors $\{S_p\}$; target exclusion fraction $\rho \in [0,1]$; floor ratio $r \in [0,1]$; optional tie-break noise on scores (default on in API)
\Ensure Binary inclusion masks $\{m_p\}$ with total kept count $k_{\mathrm{keep}} = \lfloor (1-\rho)\,N \rfloor$, $N=\sum_p \mathrm{numel}(S_p)$
\State Sanitize $S_p$ (replace NaN/Inf; dtype float32)
\If{tie-break enabled}
    \State add i.i.d.\ Gaussian noise to each $S_p$ with scale $\propto$ global max $|S|$ (fixed RNG seed in code)
\EndIf
\State Initialize all $m_p \gets 0$
\State \textbf{Floor pass:} for each $p$, set $f_p \gets \lfloor r \cdot \mathrm{numel}(S_p) \rfloor$; if $\sum_p f_p > k_{\mathrm{keep}}$, multiply all $f_p$ by $k_{\mathrm{keep}}/\sum_q f_q$ and round down within each layer’s size
\For{each $p$ with $f_p>0$}
    \State $I_p^{\mathrm{floor}} \gets$ indices of the top-$f_p$ elements of $\mathrm{vec}(S_p)$
    \State set $m_p[i]=1$ for $i \in I_p^{\mathrm{floor}}$
\EndFor
\State $k_{\mathrm{rem}} \gets k_{\mathrm{keep}} - \sum_p |I_p^{\mathrm{floor}}|$
\State \textbf{Global pass:} among positions not fixed by the floor, select the $k_{\mathrm{rem}}$ largest scores (implementation: flat $\mathrm{topk}$ on masked concatenated scores if $N$ is below an internal threshold; otherwise chunked threshold search on CPU with histogram refinement, then $\mathrm{topk}$ within the final score bin)
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{Certification threshold $\tau$ under \texttt{hybrid\_global\_phase} (global phase after per-layer floors; matches mask construction in Algorithm~\ref{alg:mask-hybrid-default})}
\label{alg:tau-hybrid-global-phase}
\begin{algorithmic}[1]
\Require Layer-partitioned selection scores $\{\tilde{S}_p\}_{p=1}^L$ in fixed iteration order; same sparsity / exclusion hyperparameters as Algorithm~\ref{alg:mask-hybrid-default}: global keep budget $k_{\mathrm{keep}}$ on $N=\sum_p \mathrm{numel}(\tilde{S}_p)$, floor ratio $r$; optional tie-break (same rule as Algorithm~\ref{alg:mask-hybrid-default})
\Ensure Scalar $\tau$ (global-phase cutoff) or undefined if the global remainder is empty
\State \textbf{Floors:} compute per-layer floor counts $\{f_p\}_{p=1}^L$ \emph{with the same scaling rule} as the floor pass of Algorithm~\ref{alg:mask-hybrid-default} (i.e.\ $f_p \approx \lfloor r\cdot \mathrm{numel}(\tilde{S}_p)\rfloor$, then scale down uniformly if $\sum_p f_p > k_{\mathrm{keep}}$, clipping each $f_p$ to $[0,\mathrm{numel}(\tilde{S}_p)]$)
\State \textbf{Sanitize and tie-break:} for each $p$, form $\mathrm{vec}(\tilde{S}_p)$; replace NaN/Inf; if tie-break is on, add i.i.d.\ Gaussian noise with scale $\propto$ global $\max |\tilde{S}|$ (fixed seed), matching Algorithm~\ref{alg:mask-hybrid-default}
\State Flatten layers in order into one vector $x \in \mathbb{R}^N$ (logical concatenation $\tilde{S}_1 \| \cdots \| \tilde{S}_L$)
\State \textbf{Floor reservation (score space):} initialize offset $o \gets 0$. For each layer $p$, let $x[o:o+n_p)$ be the slice of length $n_p=\mathrm{numel}(\tilde{S}_p)$. If $f_p>0$, let $I_p$ be the indices of the top-$f_p$ entries in that slice; set $x_i \gets -\infty$ for $i \in I_p$. Update $o \gets o+n_p$
\State Let $F \gets \sum_p f_p$ and $R \gets k_{\mathrm{keep}} - F$ \hfill \textit{// global-phase quota after floors}
\If{$R \le 0$}
    \State \Return $\tau$ undefined (NaN in code): no global-phase slots remain
\EndIf
\State Let $\mathcal{T}$ be the set of the $R$ largest \emph{finite} entries of $x$ (ties broken by the noise in step~2)
\State \Return $\tau \gets \min \mathcal{T}$ \hfill \textit{// boundary of the global-pass keep set; if $F{=}0$, $R{=}k_{\mathrm{keep}}$ recovers pure global $\tau$}
\end{algorithmic}
\end{algorithm}

\paragraph{Streaming variant (same $\tau$).}
When $N$ is too large to materialize $x$, process layers in order: for each $p$, build the sanitized/tie-broken slice, mark its top-$f_p$ entries as $-\infty$, and feed slices sequentially into a buffer that maintains only the best $R$ values seen so far over the logical concatenation; then $\tau$ is the minimum of those $R$ values (equivalently: $\tau = \min \mathrm{top}\mbox{-}R(x)$ over finite entries).

\paragraph{Relation to Algorithm~\ref{alg:mask-hybrid-default}.}
Algorithm~\ref{alg:mask-hybrid-default} assigns binary masks: floors take $F$ weights, then the global pass keeps the top-$k_{\mathrm{rem}}=R$ scores among non-floor positions. Algorithm~\ref{alg:tau-hybrid-global-phase} computes the \emph{score cutoff} $\tau$ at the bottom of that global-pass keep set. The selection scores $\{\tilde{S}_p\}$ are whatever tensor defines the ranking for the certification track (e.g.\ magnitude of weight movement $|\Delta w|$ for an oracle reference, or warm-start scores for magnitude); sanitation and tie-break match the mask pipeline so $\tau$ is comparable to the hybrid mask’s ranking semantics.

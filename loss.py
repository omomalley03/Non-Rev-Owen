import torch


def _pair_terms(F: torch.Tensor):
    """Compute sum over K^2 pairs in batch, including the plus and minus terms.
    Returns
    -------
    minus_sum, plus_sum : scalar tensors
    """
    # trace_1[k,l] — contraction over embedding dim (i) and time (t)
    trace_1 = torch.einsum("kit,lit->kl", F, F)                 # (K, K)

    # trace_2[k,l] — four-tensor contraction over i, j (embedding) and m, n (time)
    trace_2 = torch.einsum("kim,ljm,kjn,lin->kl", F, F, F, F)  # (K, K)

    return (trace_1 ** 2 - trace_2).sum(), (trace_1 ** 2 + trace_2).sum()


def S_ratio(F: torch.Tensor) -> torch.Tensor:
    """Normalised non-reversibility score, bounded ∈ [0, 1].

    S_ratio = Σ_{k,l} [Tr(G)² − Tr(G²)] / Σ_{k,l} [Tr(G)² + Tr(G²)]

    where G_{k,l} = F_k F_l^T.  Numerator and denominator both scale as
    ‖F‖⁴, so their ratio is scale-invariant — no embedding normalisation needed.

    Interpretation
    --------------
    0 → perfectly reversible (all pairs satisfy Tr(G)² = Tr(G²), i.e. G has rank 1)
    1 → maximally irreversible (all cross-trial trajectory products are maximally spread)

    This is the normalised form from Schneider et al. (SCA); we use it as
    both the training objective and the monitoring metric.
    """
    minus_sum, plus_sum = _pair_terms(F)
    return minus_sum / (plus_sum + 1e-8)


def non_reversibility_S(F: torch.Tensor) -> torch.Tensor:
    """Unnormalised non-reversibility score S (unbounded above).

    S = (2/K²) Σ_{k,l} [Tr(F_k F_l^T)² − Tr((F_k F_l^T)²)]

    Retained for reference. Prefer S_ratio for training and monitoring —
    it is scale-invariant and bounded, making lambda_bt meaningful without
    normalising F.
    """
    K = F.shape[0]
    minus_sum, _ = _pair_terms(F)
    return (2.0 / K ** 2) * minus_sum


def barlow_twins_reg(F: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Barlow Twins covariance regularizer on the per-timepoint embeddings.

    Flattens F from (K, d, T) → (M, d) where M = K*T, treating every
    (trial, timestep) pair as an independent sample. Normalises each
    embedding dimension to zero mean and unit variance, then computes the
    empirical (d, d) cross-correlation matrix Cov. Returns ‖Cov − I‖_F², which
    is zero when all dimensions are uncorrelated with unit variance.

    Internally normalises, so it is also scale-invariant. lambda_bt=5e-3
    stays meaningful alongside S_ratio ∈ [0, 1] without further tuning.
    """
    K, d, T = F.shape
    Z = F.permute(0, 2, 1).reshape(K * T, d)          # (M, d)

    Z = Z - Z.mean(dim=0, keepdim=True)               # zero-mean per dim
    Z = Z / (Z.std(dim=0, keepdim=True) + eps)        # unit-variance per dim

    M = Z.shape[0]
    # Cov[i,j] = (1/M) Σ_m Z[m,i] Z[m,j]  —  empirical cross-correlation (d, d)
    Cov = (Z.T @ Z) / M

    # Penalise deviation from identity; diagonal (variance) and off-diagonal
    # (correlation) terms treated equally, as in the original Barlow Twins paper.
    return ((Cov - torch.eye(d, device=F.device)) ** 2).sum()


def loss_fn(F: torch.Tensor, lambda_bt: float = 5e-3) -> torch.Tensor:
    """Training loss: −S_ratio(F) + λ·‖Cov − I‖_F².

    S_ratio ∈ [0, 1] is scale-invariant by construction, so lambda_bt=5e-3
    stays meaningful throughout training without any embedding normalisation.
    Set lambda_bt=0 to recover the pure S_ratio objective.
    """
    return -S_ratio(F) + lambda_bt * barlow_twins_reg(F)

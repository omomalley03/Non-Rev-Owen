import torch


def _pair_terms(F: torch.Tensor):
    """Compute sum over K^2 pairs in batch, including the plus and minus terms.
    F (aka Y) is shape (K, N, T)
    Returns
    -------
    minus_sum, plus_sum : scalar tensors
    """
    Z = torch.einsum("knt,lmt->klnm",F, F) # (K, K, N, N) -- K^2 NxN matrices 
    # trace_1[k,l] — trace of Z[k,l]

    trace_1 = torch.einsum("klnn->kl", Z)        # (K, K)

    # trace_2[k,l] - trace of Z^2
    # trace_2 = torch.einsum("kim,ljm,kjn,lin->kl", F, F, F, F)  # (K, K)
    Z_squared = torch.einsum("klnm,klmj->klnj",Z,Z) # (K, K, N, N)
    trace_2 = torch.einsum("klnn->kl",Z_squared)     # (K, K)
    # TODO: can we compute these more efficiently without forming the full (K, K, N, N) Z?
    return (trace_1 ** 2 - trace_2).sum(), (trace_1 ** 2 + trace_2).sum() 


def S_ratio(F: torch.Tensor) -> torch.Tensor:
    """Normalised non-reversibility score, bounded ∈ [0, 1].

    C^(-) is 1/K^2 sum over k,l of Tr^2(Z)-Tr(Z^2)

    """
    minus_sum, plus_sum = _pair_terms(F)
    return minus_sum / (plus_sum + 1e-8)


def non_reversibility_S(F: torch.Tensor) -> torch.Tensor:
    """
    Unnormalised non-reversibility score S .
    """
    K = F.shape[0]
    minus_sum, _ = _pair_terms(F)
    return (2.0 / K ** 2) * minus_sum


def barlow_twins_reg(F: torch.Tensor, eps: float = 1e-6, normalize: bool = False) -> torch.Tensor:
    """Barlow Twins covariance regularizer on the per-timepoint embeddings.

    Flattens F from (K, d, T) → (M, d) where M = K*T, treating every
    (trial, timestep) pair as an independent sample. Computes the
    empirical (d, d) Cov. Returns ‖Cov - I‖_F², which
    is zero when all dimensions are uncorrelated with unit variance.

    """
    K, d, T = F.shape
    Z = F.permute(0, 2, 1).reshape(K * T, d)          # (M, d)

    # if normalize:
    #     Z = Z - Z.mean(dim=0, keepdim=True)               # zero-mean per dim
    #     Z = Z / (Z.std(dim=0, keepdim=True) + eps)        # unit-variance per dim

    M = Z.shape[0]

    Cov = (Z.T @ Z) / M

    return ((Cov - torch.eye(d, device=F.device)) ** 2).sum()


def _batch_rms_normalize(F: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Divide F by a single batch-wide scalar r = sqrt(mean_k |F_k|_F^2). 

    All K trials rescaled by same r, so every pair term inside the
    batch is rescaled identically by 1/r^4.  
    """
    mean_sq_norm = F.pow(2).sum(dim=(1, 2)).mean()    # scalar 
    return F / (mean_sq_norm + eps).sqrt() # divide by RMS across the batch


# def _detrend(F: torch.Tensor) -> torch.Tensor:
#     """Project out the linear trend (constant + ramp) from each trial's embedding."""
#     T = F.shape[2]
#     e0 = torch.ones(T, dtype=F.dtype, device=F.device)
#     e0 = e0 / e0.norm()
#     e1 = torch.arange(T, dtype=F.dtype, device=F.device)
#     e1 = e1 - e1.mean()
#     e1 = e1 / e1.norm()
#     basis = torch.stack([e0, e1])                        # (2, T)
#     return F - (F @ basis.T) @ basis                     # (K, d, T)


def _detrend(F: torch.Tensor) -> torch.Tensor:
    """Subtract the best-fit line (a + b·t) from each embedding dimension per trial."""
    T = F.shape[2]
    t = torch.arange(T, dtype=F.dtype, device=F.device) - (T - 1) / 2  # zero-mean time axis

    intercept = F.mean(dim=2, keepdim=True)                        # (K, d, 1)
    slope = (F * t).sum(dim=2, keepdim=True) / (t * t).sum()      # (K, d, 1)

    return F - intercept - slope * t                               # (K, d, T)


def loss_fn(F: torch.Tensor, lambda_bt: float = 5e-3, normalize_bt: bool = False) -> torch.Tensor:
    """Training loss: −S(detrend(F̂)) + λ·BT(F).

    Detrending projects out the linear ramp from each trial's embedding
    before computing S, so the score only sees oscillatory (rotational)
    structure.  Unlike temporal differentiation, detrending does not
    amplify noise.
    """
    F_dt = _detrend(F)
    # F_dt = F
    F_dt_hat = _batch_rms_normalize(F_dt)
    return -non_reversibility_S(F_dt_hat) + lambda_bt * barlow_twins_reg(F, normalize=normalize_bt)



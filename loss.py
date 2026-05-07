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
    trace_2 = torch.einsum("klnn->kl",Z_squared)

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


def barlow_twins_reg(F: torch.Tensor, eps: float = 1e-6, normalize: bool = True) -> torch.Tensor:
    """Barlow Twins covariance regularizer on the per-timepoint embeddings.

    Flattens F from (K, d, T) → (M, d) where M = K*T, treating every
    (trial, timestep) pair as an independent sample. Normalises each
    embedding dimension to zero mean and unit variance, then computes the
    empirical (d, d) cross-correlation matrix Cov. Returns ‖Cov - I‖_F², which
    is zero when all dimensions are uncorrelated with unit variance.

    Internally normalises, so it is also scale-invariant. lambda_bt=5e-3
    stays meaningful alongside S_ratio ∈ [0, 1] without further tuning.
    """
    K, d, T = F.shape
    Z = F.permute(0, 2, 1).reshape(K * T, d)          # (M, d)
    if normalize:
        Z = Z - Z.mean(dim=0, keepdim=True)               # zero-mean per dim
        Z = Z / (Z.std(dim=0, keepdim=True) + eps)        # unit-variance per dim

    M = Z.shape[0]

    Cov = (Z.T @ Z) / M

    return ((Cov - torch.eye(d, device=F.device)) ** 2).sum()


def _batch_rms_normalize(F: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Divide F by a single batch-wide scalar r = sqrt(mean_k |F_k|_F^2). 

    All K trials rescaled by same r, so every pair term inside the
    batch is rescaled identically by 1/r^4.  
    """
    mean_sq_norm = F.pow(2).sum(dim=(1, 2)).mean()    # scalar 
    return F / (mean_sq_norm + eps).sqrt()


def loss_fn(F: torch.Tensor, lambda_bt: float = 5e-3, normalize_bt: bool = True) -> torch.Tensor:
    """Training loss: −S(F̂) + λ·‖Cov(F̂) − I‖_F², with F̂ = F / batch_RMS.

    Batch-RMS normalisation bounds the loss magnitude (otherwise both terms
    scale as ‖F‖⁴ and diverge) without introducing the per-trial bias that
    plagues S_ratio: every trial in the batch is divided by the same scalar,
    so pair-wise gradients keep their correct relative weights.
    """
    F_hat = _batch_rms_normalize(F)
    return -non_reversibility_S(F_hat) + lambda_bt * barlow_twins_reg(F_hat, normalize=normalize_bt)

if __name__=="__main__":
    torch.manual_seed(42)
    F = torch.rand((1,2,3))
    print(F)
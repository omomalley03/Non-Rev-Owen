import torch


def _pair_terms(F: torch.Tensor):
    """Compute sum over K^2 pairs in batch, per 2D rotation plane.

    F is shape (K, 2D, T) where 2D = d must be even.
    Each consecutive pair of dimensions forms one rotation plane.

    Returns
    -------
    minus_sum, plus_sum : scalar tensors  (summed over all planes and pairs)
    """
    K, d, T = F.shape
    assert d % 2 == 0, f"d must be even for per-plane decomposition, got {d}"
    D = d // 2

    F_p = F.reshape(K, D, 2, T)                                  # (K, D, 2, T)

    Z = torch.einsum("kpit,lpmt->klpim", F_p, F_p)               # (K, K, D, 2, 2)
    trace_1 = torch.einsum("klpii->klp", Z)                      # (K, K, D)

    Z_squared = torch.einsum("klpim,klpmj->klpij", Z, Z)         # (K, K, D, 2, 2)
    trace_2 = torch.einsum("klpii->klp", Z_squared)              # (K, K, D)

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


def non_rev_regularizer(F: torch.Tensor) -> torch.Tensor:
    """Regularize by minimizing cross-plane non-reversibility score"""
    F_shuff = F[F.randperm(F.shape[0])]  # shuffle batch dimension to break within-plane structure
    return non_reversibility_S(F_shuff)

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
    Z = Z-Z.mean(dim=0, keepdims=True)               # zero-mean per dim
    Cov = (Z.T @ Z) / M

    return ((Cov - torch.eye(d, device=F.device)) ** 2).sum()


def _batch_rms_normalize(F: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-plane RMS normalization: each 2D plane gets its own scalar."""
    K, d, T = F.shape
    D = d // 2
    F_p = F.reshape(K, D, 2, T)                          # (K, D, 2, T)
    sq_norms = F_p.pow(2).sum(dim=(2, 3))                 # (K, D)
    mean_sq = sq_norms.mean(dim=0)                         # (D,)
    rms = (mean_sq + eps).sqrt().reshape(1, D, 1, 1)       # (1, D, 1, 1)
    return (F_p / rms).reshape(K, d, T)


def loss_fn(F: torch.Tensor, lambda_bt: float = 5e-3, normalize_bt: bool = False) -> torch.Tensor:
    """Training loss: −S(F̂) + λ·BT(F).

    S is computed per 2D rotation plane on RMS-normalised embeddings.
    BT decorrelates across all d dimensions.
    """
    # F = F-F.mean(dim=cfg.F_mean_axis, keepdims=True)  # zero-mean per dim across batch and time
    F_hat = _batch_rms_normalize(F)
    return -non_reversibility_S(F_hat) + lambda_bt * barlow_twins_reg(F, normalize=normalize_bt)
    # return lambda_bt * barlow_twins_reg(F, normalize=normalize_bt)

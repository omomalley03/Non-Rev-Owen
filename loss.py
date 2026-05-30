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
    idx = torch.randperm(F.shape[1])
    F_shuff = F[:, idx, :]
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


def plane_barlow_twins_reg(F: torch.Tensor) -> torch.Tensor:
    """Plane-aware Barlow Twins: penalise cross-plane covariance, allow within-plane.

    Like barlow_twins_reg but the within-plane off-diagonal entries (e.g.
    Cov[0,1] and Cov[1,0] for plane 0) are masked out of the penalty.
    Rotation naturally creates within-plane correlation, so penalising it
    would fight the primary loss.

    Diagonal entries (variance → 1) and all cross-plane entries are still
    penalised as in standard Barlow Twins.

    Returns ‖(Cov - I) ⊙ mask‖_F² where mask zeros within-plane off-diag.
    """
    K, d, T = F.shape
    D = d // 2
    Z = F.permute(0, 2, 1).reshape(K * T, d)
    M = Z.shape[0]
    Z = Z - Z.mean(dim=0, keepdim=True)
    Cov = (Z.T @ Z) / M

    diff = Cov - torch.eye(d, device=F.device)

    # mask: True = penalised, False = allowed (within-plane off-diagonal)
    mask = torch.ones(d, d, dtype=torch.bool, device=F.device)
    for p in range(D):
        mask[2*p, 2*p + 1] = False
        mask[2*p + 1, 2*p] = False

    return (diff[mask] ** 2).sum()


def _batch_rms_normalize(F: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-plane RMS normalization: each 2D plane gets its own scalar."""
    K, d, T = F.shape
    D = d // 2
    F_p = F.reshape(K, D, 2, T)                          # (K, D, 2, T)
    sq_norms = F_p.pow(2).sum(dim=(2, 3))                 # (K, D)
    mean_sq = sq_norms.mean(dim=0)                         # (D,)
    rms = (mean_sq + eps).sqrt().reshape(1, D, 1, 1)       # (1, D, 1, 1)
    return (F_p / rms).reshape(K, d, T)


def loss_fn(F: torch.Tensor, lambda_xp: float = 1.0, lambda_bt: float = 1.0) -> torch.Tensor:
    """Training loss: −S(F̂) + λ_xp·cross_plane_reg(F̂) + λ_bt·BT(F).

    S and cross-plane reg are computed on RMS-normalised embeddings.
    BT operates on the raw embeddings.
    """
    d = F.shape[1]
    planes = d // 2

    F_hat = _batch_rms_normalize(F)
    
    non_rev_reg = 0
    if lambda_xp > 0:

        for _ in range(planes//2):
            non_rev_reg += non_rev_regularizer(F_hat) # now scales with number of planes and expected number for each cross-planes is 1
    
    return (-non_reversibility_S(F_hat)
            + lambda_xp * non_rev_reg
            + lambda_bt * barlow_twins_reg(F)
    )
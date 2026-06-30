from __future__ import annotations

import torch

from config import Config


def _pair_terms_per_plane(F: torch.Tensor):
    """Compute sum over K^2 pairs in batch, per 2D rotation plane.

    F is shape (K, 2D, T) where 2D = d must be even.
    Each consecutive pair of dimensions forms one rotation plane.

    Returns
    -------
    minus_sum, plus_sum : tensors of shape (D,), summed over all K^2 pairs.
    """
    K, d, T = F.shape
    assert d % 2 == 0, f"d must be even for per-plane decomposition, got {d}"
    D = d // 2

    F_p = F.reshape(K, D, 2, T)                                  # (K, D, 2, T)

    Z = torch.einsum("kpit,lpmt->klpim", F_p, F_p)               # (K, K, D, 2, 2)
    trace_1 = torch.einsum("klpii->klp", Z)                      # (K, K, D)

    Z_squared = torch.einsum("klpim,klpmj->klpij", Z, Z)         # (K, K, D, 2, 2)
    trace_2 = torch.einsum("klpii->klp", Z_squared)              # (K, K, D)

    return (trace_1 ** 2 - trace_2).sum(dim=(0, 1)), (trace_1 ** 2 + trace_2).sum(dim=(0, 1))


def _pair_terms(F: torch.Tensor):
    """Compute pair terms summed over all 2D planes."""
    minus_per_plane, plus_per_plane = _pair_terms_per_plane(F)
    return minus_per_plane.sum(), plus_per_plane.sum()

def _DEPRECATED_pair_terms(F: torch.Tensor):
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


def non_reversibility_S(F: torch.Tensor, objective: str = "sum") -> torch.Tensor:
    """
    Unnormalised non-reversibility score S .
    """
    K = F.shape[0]
    minus_sum, _ = _pair_terms(F)
    if objective == "mean":
        return (4.0 / K ** 2) * minus_sum / F.shape[1]  # divide by d to keep scale comparable across dimensions
    return (2.0 / K ** 2) * minus_sum
    # regular non_rev_S / (d/2) divide by number of planes to keep scale comparable across dimensions, since more planes means more pairs and higher expected non-rev score

def non_reversibility_S_per_plane(F: torch.Tensor) -> torch.Tensor:
    """Unnormalised non-reversibility score for each 2D plane."""
    K = F.shape[0]
    minus_per_plane, _ = _pair_terms_per_plane(F)
    return (2.0 / K ** 2) * minus_per_plane


def non_rev_regularizer(F: torch.Tensor, cfg: Config) -> torch.Tensor: # randomize dim order and calculate non-rev score cross-plane
    """Regularize by minimizing cross-plane non-reversibility score.

    Always uses objective="mean" regardless of cfg.s_objective — the xp penalty
    measures average cross-plane structure; using softmin here conflicts with the
    softmin main objective and causes embedding collapse.
    """
    idx = torch.randperm(F.shape[1])
    F_shuff = F[:, idx, :]
    return non_reversibility_S(F_shuff, objective="mean")


def non_rev_regularizer_systematic(F: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Systematic (exhaustive) cross-plane non-reversibility regularizer.

    non_rev_regularizer scores a single random re-pairing of dims, so any given
    cross-plane pair is only penalised in expectation. This instead enumerates
    every dimension pair (i, j), i < j, that is NOT a native rotation plane --
    i.e. all pairs except (0,1), (2,3), ... -- forms a 2D plane from each, and
    sums their non-reversibility scores. Driving it down suppresses rotation
    structure between dims belonging to different planes.
    """
    d = F.shape[1]
    native = {(2 * p, 2 * p + 1) for p in range(d // 2)}
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d) if (i, j) not in native]
    idx = torch.tensor([k for pair in pairs for k in pair], device=F.device)
    return non_reversibility_S(F[:, idx, :], objective=cfg.s_objective)


def barlow_twins_reg(F: torch.Tensor, eps: float = 1e-6, normalize: bool = False) -> torch.Tensor:
    """Barlow Twins covariance regularizer on the per-timepoint embeddings.

    Flattens F from (K, d, T) → (M, d) where M = K*T, treating every
    (trial, timestep) pair as an independent sample. Computes the
    empirical (d, d) Cov. Returns mean((Cov - I)²) normalised by d²,
    so lambda_bt has the same interpretation regardless of d.
    """
    K, d, T = F.shape
    Z = F.permute(0, 2, 1).reshape(K * T, d)          # (M, d)

    M = Z.shape[0]
    Z = Z - Z.mean(dim=0, keepdim=True)               # zero-mean per dim
    Cov = (Z.T @ Z) / M

    return ((Cov - torch.eye(d, device=F.device)) ** 2).sum() / (d * d)


def plane_barlow_twins_reg(F: torch.Tensor) -> torch.Tensor:
    """Plane-aware Barlow Twins: penalise cross-plane covariance, allow within-plane.

    Like barlow_twins_reg but the within-plane off-diagonal entries (e.g.
    Cov[0,1] and Cov[1,0] for plane 0) are masked out of the penalty.
    Rotation naturally creates within-plane correlation, so penalising it
    would fight the primary loss.

    Diagonal entries (variance → 1) and all cross-plane entries are still
    penalised as in standard Barlow Twins.

    Returns mean((Cov - I)²) over penalised entries, normalised by d²
    so lambda_plane_bt is comparable to lambda_bt across different d.
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

    return (diff[mask] ** 2).sum() / (d * d)


def _batch_rms_normalize(F: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-plane RMS normalization: each 2D plane gets its own scalar."""
    K, d, T = F.shape
    D = d // 2
    F_p = F.reshape(K, D, 2, T)                          # (K, D, 2, T)
    sq_norms = F_p.pow(2).sum(dim=(2, 3))                 # (K, D)
    mean_sq = sq_norms.mean(dim=0)                         # (D,)
    rms = (mean_sq + eps).sqrt().reshape(1, D, 1, 1)       # (1, D, 1, 1)
    return (F_p / rms).reshape(K, d, T)


def _plane_samples(F: torch.Tensor) -> torch.Tensor:
    """Return plane snapshots with shape (D, M, 2)."""
    K, d, T = F.shape
    D = d // 2
    return F.reshape(K, D, 2, T).permute(1, 0, 3, 2).reshape(D, K * T, 2)


def _whiten_2d(X: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Whiten each 2D plane in a batch, shape (D, M, 2)."""
    X = X - X.mean(dim=1, keepdim=True)
    cov = torch.einsum("dmi,dmj->dij", X, X) / max(X.shape[1] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    inv_sqrt = eigvecs @ torch.diag_embed(torch.rsqrt(eigvals.clamp_min(eps))) @ eigvecs.transpose(-1, -2)
    return torch.einsum("dmi,dij->dmj", X, inv_sqrt)


def block_cca_reg(F: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Penalize linear dependence between whole 2D planes."""
    X = _whiten_2d(_plane_samples(F), eps=eps)             # (D, M, 2)
    D, M, _ = X.shape
    if D < 2:
        return F.new_tensor(0.0)
    C = torch.einsum("pmi,qmj->pqij", X, X) / M            # (D, D, 2, 2)
    eye = torch.eye(D, device=F.device, dtype=torch.bool)
    return C[~eye].pow(2).sum() / (D * (D - 1))


# Regularizer registry: name -> (config lambda attr, raw-magnitude function).
# `xp` (cross-plane non-rev) needs cfg, so it is handled separately below.
REG_TYPES = ("xp", "bt", "plane_bt", "cca")


def loss_fn(F: torch.Tensor, cfg=None, lambda_xp: float | None = None, lambda_bt: float | None = None,
            training: bool = True, lambda_scale: float = 1.0, return_components: bool = False):
    """Training loss with independently weighted plane regularizers.

    S and cross-plane reg are computed on RMS-normalised embeddings.
    BT and redundancy terms operate on the raw embeddings unless noted.

    `lambda_scale` ∈ [0, 1] uniformly scales every regularizer weight (used by
    the linear lambda warm-up in train.py).  When `return_components=True`,
    returns `(loss, info)` where info["reg_raw"] holds each regularizer's raw
    (unscaled) magnitude and info["reg_scaled"] holds `lambda_scale·λ·raw`.
    """
    if cfg is not None:
        lambda_xp = cfg.lambda_xp
        lambda_bt = cfg.lambda_bt
    lambda_xp = 0.0 if lambda_xp is None else lambda_xp
    lambda_bt = 0.1 if lambda_bt is None else lambda_bt
    weights = {
        "xp": lambda_xp,
        "bt": lambda_bt,
        "plane_bt": getattr(cfg, "lambda_plane_bt", 0.0),
        "cca": getattr(cfg, "lambda_block_cca", 0.0),
    }

    F_hat = _batch_rms_normalize(F)

    if getattr(cfg, "s_objective", "sum") == "softmin":
        S_p = non_reversibility_S_per_plane(F_hat)
        tau = max(float(getattr(cfg, "s_softmin_tau", 0.1)), 1e-6)
        loss = tau * torch.logsumexp(-S_p / tau, dim=0)
    else:
        loss = -non_reversibility_S(F_hat, objective=getattr(cfg, "s_objective", "sum"))

    # raw regularizer magnitudes (unscaled by lambda); only compute active ones
    raw = {}
    if weights["xp"] > 0:
        raw["xp"] = non_rev_regularizer(F_hat, cfg)
    if weights["bt"] > 0:
        raw["bt"] = barlow_twins_reg(F)
    if weights["plane_bt"] > 0:
        raw["plane_bt"] = plane_barlow_twins_reg(F)
    if weights["cca"] > 0:
        raw["cca"] = block_cca_reg(F)

    reg = F.new_tensor(0.0)
    for name, value in raw.items():
        reg = reg + lambda_scale * weights[name] * value

    loss = loss + reg

    if return_components:
        info = {
            "reg_raw":    {k: v.item() for k, v in raw.items()},
            "reg_scaled": {k: (lambda_scale * weights[k] * raw[k]).item() for k in raw},
            "lambda_scale": lambda_scale,
        }
        return loss, info
    return loss

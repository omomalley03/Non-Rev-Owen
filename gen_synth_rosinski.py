"""
Port of the SCA_project toy-data generator (Rosinski et al.) into this repo.

This is a faithful NumPy re-implementation of the `rotation` pipeline in
SCA_project/run_toy_datasets.py.  Unlike our other synthetic generators it
produces CONTINUOUS, rate-like data (no spikes); the distinguishing feature is
the realistic noise model.

Pipeline
--------
  1. Ground-truth latent: a 2D rotation per trial.  Radius (0.1-2, permuted) and
     phase (linspace 0-2π) vary across trials, so trials differ in amplitude and
     starting angle.  Shape (K, 2, T).
  2. Lift to N channels via a random ORTHONORMAL projection (QR'd), which
     preserves the latent geometry exactly.
  3. Corrupt with LOW-RANK, TEMPORALLY-CORRELATED noise: NOISE_RANK random
     orthonormal directions in channel space, each carrying a smooth time course
     drawn from a squared-exponential GP (lengthscale L2).  This models shared
     neural variability far better than i.i.d. noise.  An isotropic variant is
     also available.

Output: the noisy projected data, shape (K, T, N), saved as synth_rosinski.npy
(matching load_synthetic_windows, which transposes to (K, N, T)).  The clean 2D
latent is saved alongside as synth_rosinski_latent.npy for reference.
"""

import numpy as np

K = 200               # trials
T = 100                # timesteps
N = 50                 # channels (= projection target dim)
NOISE_RANK = 3        # rank of the correlated noise
SIGMA_LOW_RANK = 0.75  # noise amplitude
L2 = 0.1              # squared-exponential GP lengthscale over time


def get_rotations(rng: np.random.Generator) -> np.ndarray:
    """Ground-truth 2D rotation latent, shape (K, 2, T)."""
    time = np.linspace(0, 2 * np.pi, T)[:, None]      # (T, 1)
    radii = rng.permutation(np.linspace(0.1, 2, K))   # (K,)
    phases = np.linspace(0, 2 * np.pi, K)             # (K,)

    cosine = np.cos(time + phases) * radii            # (T, K)
    sine = np.sin(time + phases) * radii              # (T, K)
    return np.stack([cosine.T, sine.T], axis=1)       # (K, 2, T)


def project_X(X: np.ndarray, rng: np.random.Generator, n_channels: int) -> np.ndarray:
    """Lift (K, d, T) → (K, n_channels, T) through a random orthonormal map."""
    P = rng.standard_normal((n_channels, X.shape[1]))  # (N, d)
    Q, _ = np.linalg.qr(P)                             # (N, d), orthonormal columns
    return np.einsum("lj,ijk->ilk", Q, X)             # (K, N, T)


def _time_cov(rank_T: int, l2: float) -> np.ndarray:
    """Squared-exponential covariance over T evenly spaced time points."""
    t = np.linspace(0, 1, rank_T)
    sq = (t[:, None] - t[None, :]) ** 2
    return np.exp(-0.5 / l2 * sq)


def add_low_rank_noise(X: np.ndarray, rng: np.random.Generator,
                       rank: int, sigma: float, l2: float) -> np.ndarray:
    """Add low-rank, temporally-correlated GP noise to (K, N, T) data."""
    Kk, Nn, Tt = X.shape
    B = rng.standard_normal((Nn, rank))
    B, _ = np.linalg.qr(B)                             # (N, rank) orthonormal directions

    L = np.linalg.cholesky(_time_cov(Tt, l2) + np.eye(Tt) * 1e-5)   # (T, T)
    eps = rng.standard_normal((Kk, Tt, rank)) * sigma              # (K, T, rank)
    eps = np.einsum("ts,ksd->ktd", L, eps)                         # correlate in time
    noise = np.einsum("ktd,nd->knt", eps, B)                       # spread over channels
    return X + noise


def add_isotropic_noise(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add tiny i.i.d. Gaussian noise to (K, N, T) data."""
    return X + 0.01 * rng.standard_normal(X.shape)


def generate(seed: int = 42, noise: str = "low_rank"):
    """Return (neural (K, N, T), clean latent (K, 2, T))."""
    rng = np.random.default_rng(seed)

    latent = get_rotations(rng)                       # (K, 2, T)
    X = project_X(latent, rng, N)                     # (K, N, T)

    if noise == "low_rank":
        X = add_low_rank_noise(X, rng, NOISE_RANK, SIGMA_LOW_RANK, L2)
    elif noise == "isotropic":
        X = add_isotropic_noise(X, rng)
    elif noise != "none":
        raise ValueError(f"unknown noise type: {noise!r}")

    return X.astype(np.float32), latent.astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", choices=["low_rank", "isotropic", "none"],
                        default="low_rank")
    parser.add_argument("--out", type=str, default="synth_rosinski.npy")
    args = parser.parse_args()

    X, latent = generate(args.seed, args.noise)

    # Save in (K, T, N) convention expected by load_synthetic_windows.
    np.save(args.out, X.transpose(0, 2, 1))
    latent_out = args.out.replace(".npy", "_latent.npy")
    np.save(latent_out, latent.transpose(0, 2, 1))    # (K, T, 2) ground truth

    snr = latent.var() / max(X.var() - latent.var(), 1e-12)
    print(f"Saved {X.shape} (K, N, T) → {args.out}  [noise={args.noise}]")
    print(f"  ground-truth 2D latent → {latent_out}")
    print(f"  value range: [{X.min():.2f}, {X.max():.2f}]  mean {X.mean():.3f}")

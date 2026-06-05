"""
Generate synthetic rotational latent data.

  k=200 trials, t=100 timesteps, o=128 output dimensions.

Each trial has 4 independent 2D rotations with DIFFERENT frequencies
(default: 1, 2, 3, 4 full revolutions over T steps) and per-trial random
phase offsets, projected to o dimensions via a fixed Gaussian mixing matrix.

Using distinct frequencies is essential: it makes the sources separable —
a linear mix of different-frequency rotations produces a Lissajous figure,
not a circle, so the S-loss can only score well by routing each frequency to
its own output plane. With identical frequencies (old behaviour) any mixture
is still a circle, the sources are indistinguishable, and the model produces
concentric rings from phase interference rather than cleanly separated planes.

Output shape: (k, t, o)  →  saved as rotations_v6.npy
"""

import numpy as np

K = 200
T = 100
O = 128
# Frequencies must have irrational pairwise ratios so cross-plane Lissajous
# figures are space-filling (not closed curves).  Integer or simple-fraction
# ratios produce structured closed curves that look like ellipses in the grid
# plot even when the model has correctly separated the sources.
# Golden-ratio spacing: 1, φ, φ², φ³  (φ ≈ 1.618) guarantees every pairwise
# ratio is an irrational power of φ, avoiding all harmonic relationships.
_phi = (1 + np.sqrt(5)) / 2
planes = 8
FREQS = [_phi**i for i in range(planes)]   # ≈ [1.0, 1.618, 2.618, 4.236]


def latent_rotations(rng: np.random.Generator, freq: float = 1.0) -> np.ndarray:
    """Return (K, T, 2): cosine/sine pair at given frequency with random phase."""
    phases = rng.uniform(0, 2 * np.pi, (K, 1))              # (K, 1)
    ts = np.linspace(0, 2 * np.pi * freq, T)                 # (T,)
    x = np.cos(ts + phases).reshape(K, T, 1)
    y = np.sin(ts + phases).reshape(K, T, 1)
    return np.concatenate([x, y], axis=2)                    # (K, T, 2)


def generate(seed: int = 10, freqs: list[float] = FREQS) -> np.ndarray:
    rng = np.random.default_rng(seed)

    n_sources = len(freqs)
    c = rng.standard_normal((2 * n_sources, O)) / np.sqrt(2 * n_sources)  # mixing matrix

    # 4 independent rotation pairs at different frequencies → (K*T, 2*n_sources)
    latent_rots = np.concatenate(
        [latent_rotations(rng, freq=f).reshape(-1, 2) for f in freqs],
        axis=1,
    )

    return (latent_rots @ c).reshape(K, T, O)                # (K, T, O)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="rotations_8planes.npy")
    args = parser.parse_args()

    data = generate(args.seed)
    np.save(args.out, data)
    print(f"Saved {data.shape} → {args.out}")

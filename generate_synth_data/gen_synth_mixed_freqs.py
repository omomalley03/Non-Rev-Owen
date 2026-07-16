"""
Generate synthetic rotational latent data with multiple frequencies mixed
inside each latent 2D rotation plane before projection.

Each latent plane is a weighted sum of several complex rotations. Negative
frequencies rotate in the opposite direction.

Example default:
  each latent plane has:
    - one low-frequency CCW component
    - one higher-frequency CW component

Output shape: (K, T, O) -> saved as rotations_mixed_freqs.npy
"""

import argparse
import numpy as np

K = 1024
T = 100
O = 128

_phi = (1 + np.sqrt(5)) / 2

# Number of low-dimensional latent 2D planes before projecting to O dimensions.
# With 4 planes, latent dimension is 8, then projected to 128.
N_LATENT_PLANES = 4

# Each entry is one latent 2D plane.
# Each tuple is: (frequency, amplitude)
#
# Positive frequency = CCW
# Negative frequency = CW
#
# So this creates high-frequency CW motion inside lower-frequency CCW motion.
FREQ_GROUPS = [
    [
        (_phi**p, 1.0),          # low-frequency CCW component
        (-4.0 * _phi**p, 0.35),  # higher-frequency CW component
    ]
    for p in range(N_LATENT_PLANES)
]


def latent_mixed_rotation(
    rng: np.random.Generator,
    components: list[tuple[float, float]],
) -> np.ndarray:
    """
    Return one mixed-frequency latent 2D plane.

    Parameters
    ----------
    components:
        List of (frequency, amplitude) pairs.

    Returns
    -------
    xy : np.ndarray
        Shape (K, T, 2). The final dimension is [x, y].
    """
    tau = np.linspace(0.0, 1.0, T, endpoint=False)  # normalized time in [0, 1)

    z = np.zeros((K, T), dtype=np.complex128)

    for freq, amp in components:
        phases = rng.uniform(0, 2 * np.pi, size=(K, 1))
        theta = 2 * np.pi * freq * tau[None, :] + phases
        z += amp * np.exp(1j * theta)

    xy = np.stack([z.real, z.imag], axis=2)  # (K, T, 2)

    # Normalize each latent plane so planes with more components do not dominate.
    xy = xy / (np.sqrt(np.mean(xy**2)) + 1e-12)

    return xy


def generate(
    seed: int = 0,
    freq_groups: list[list[tuple[float, float]]] = FREQ_GROUPS,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    n_sources = len(freq_groups)
    latent_dim = 2 * n_sources

    # Fixed Gaussian mixing matrix: latent_dim -> O
    c = rng.standard_normal((latent_dim, O)) / np.sqrt(latent_dim)

    # Build mixed-frequency latent rotations.
    # Each source is (K, T, 2), then flattened to (K*T, 2).
    latent_rots = np.concatenate(
        [
            latent_mixed_rotation(rng, components).reshape(-1, 2)
            for components in freq_groups
        ],
        axis=1,
    )  # (K*T, 2*n_sources)

    data = latent_rots @ c  # (K*T, O)

    return data.reshape(K, T, O)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="rotations_mixed_freqs.npy")
    args = parser.parse_args()

    data = generate(seed=args.seed)
    np.save(args.out, data)

    print(f"Saved {data.shape} -> {args.out}")
    print(f"Latent planes: {len(FREQ_GROUPS)}")
    print(f"Latent dimension before projection: {2 * len(FREQ_GROUPS)}")
    print(f"Projected dimension: {O}")
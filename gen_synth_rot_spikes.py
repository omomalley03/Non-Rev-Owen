"""
Binary synthetic spikes built from a fixed number of ground-truth rotations.

Idea
----
1. Fix ROTATIONS = R, the ground-truth number of 2D rotations.  Each rotation r
   is a circular latent (cos/sin) at its own frequency with a per-trial phase:
       L_r(t) = [cos(2π f_r t + φ), sin(2π f_r t + φ)]
   stacked over r → 2R latent coordinates per (trial, time).

2. Each output channel is a FIXED random linear mix of those 2R coordinates,
   so every channel shares the same R underlying rotations (just weighted
   differently).  The mix is a 1D oscillating signal.

3. Binary readout: a channel spikes wherever its signal crosses zero — i.e. it
   intersects the zero axis.  Same "intersection → spike" rule as before.

Different frequencies (golden-ratio spaced, so no harmonic ratios) keep the
rotations separable.  Output: (K, T, O) binary array → saved as
synth_rot_spikes.npy.
"""

import numpy as np

K = 200          # trials
T = 90           # timesteps
O = 137          # channels
ROTATIONS = 4    # <-- fixed ground-truth number of rotations

# Golden-ratio frequency spacing: every pairwise ratio is irrational, so the
# rotations stay separable (no closed Lissajous figures).  Units = revolutions
# over the window; kept modest so zero-crossings stay reasonably sparse.
_phi = (1 + np.sqrt(5)) / 2
FREQS = [_phi**i for i in range(ROTATIONS)]   # ≈ [1.0, 1.62, 2.62, 4.24] for R=4


def rotation_latents(rng: np.random.Generator) -> np.ndarray:
    """Return (K, T, 2R): R cos/sin pairs at FREQS with per-trial random phase."""
    t = np.linspace(0.0, 1.0, T)                          # (T,)
    pairs = []
    for f in FREQS:
        phase = rng.uniform(0, 2 * np.pi, (K, 1))         # (K, 1) per-trial phase
        ang = 2 * np.pi * f * t[None, :] + phase          # (K, T)
        pairs.append(np.stack([np.cos(ang), np.sin(ang)], axis=2))   # (K, T, 2)
    return np.concatenate(pairs, axis=2)                  # (K, T, 2R)


def generate(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    latents = rotation_latents(rng)                       # (K, T, 2R)
    C = rng.standard_normal((2 * ROTATIONS, O)) / np.sqrt(2 * ROTATIONS)  # fixed mix

    signal = latents @ C                                  # (K, T, O) continuous

    crossed = np.sign(signal)[:, 1:] != np.sign(signal)[:, :-1]   # (K, T-1, O)
    spikes = np.zeros((K, T, O), dtype=np.float32)
    spikes[:, 1:][crossed] = 1.0
    return spikes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="synth_rot_spikes.npy")
    args = parser.parse_args()

    data = generate(args.seed)
    np.save(args.out, data)

    print(f"Saved {data.shape} → {args.out}  ({ROTATIONS} ground-truth rotations)")
    print(f"  binary values: {np.unique(data).tolist()}")
    print(f"  sparsity: {(data == 0).mean():.3%} zeros  |  "
          f"mean {data.mean():.4f} spikes/bin  |  "
          f"{data.sum(1).mean():.2f} spikes/trial/channel")

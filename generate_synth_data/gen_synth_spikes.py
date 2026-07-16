"""
Generate synthetic spike-count data in MC_Maze format.

Same ground-truth structure as gen_synth.py (8 independent 2D rotation planes at
golden-ratio frequencies, projected to the output dimensions by a fixed Gaussian
mixing matrix), but instead of using the continuous mix directly we treat it as a
log-firing-rate, scale it to MC_Maze firing statistics, and Poisson-sample
integer spike counts.

The result is sparse non-negative count data in the same format as the MC_Maze
spikes (137 channels, 10 ms bins, ~97% zeros, small integer counts), which
main_synth.py then runs through the SAME gaussian_smooth + soft_normalize
preprocessing as the real data.

Output shape: (K, T, O) integer spike counts → saved as synth_spikes_8planes.npy
"""

import numpy as np

K = 200       # trials
T = 90        # timesteps per trial (matches MC_Maze window_size)
O = 137       # channels (matches MC_Maze)

# Golden-ratio frequency spacing — every pairwise ratio is irrational, so
# cross-plane Lissajous figures are space-filling and the sources are separable.
_phi = (1 + np.sqrt(5)) / 2
PLANES = 8
FREQS = [_phi**i for i in range(PLANES)]

# MC_Maze (Jenkins) per-neuron mean spike-count/bin at 10 ms is roughly
# lognormal: median ~0.017, spanning ~0.001–0.15.  Draw per-neuron baseline
# rates from a matched lognormal, then let the rotational latent modulate each
# neuron's log-rate around its baseline.
BASE_LOG_MEAN = np.log(0.017)
BASE_LOG_STD = 0.9
GAIN = 1.0    # modulation depth of the latent on the log firing rate


def latent_rotations(rng: np.random.Generator, freq: float) -> np.ndarray:
    """Return (K, T, 2): cosine/sine pair at given frequency with random phase."""
    phases = rng.uniform(0, 2 * np.pi, (K, 1))
    ts = np.linspace(0, 2 * np.pi * freq, T)
    x = np.cos(ts + phases).reshape(K, T, 1)
    y = np.sin(ts + phases).reshape(K, T, 1)
    return np.concatenate([x, y], axis=2)


def generate(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    n_sources = len(FREQS)
    c = rng.standard_normal((2 * n_sources, O)) / np.sqrt(2 * n_sources)  # mixing matrix

    latents = np.concatenate(
        [latent_rotations(rng, f).reshape(-1, 2) for f in FREQS], axis=1
    )                                            # (K*T, 2*n_sources)
    z = (latents @ c).reshape(K, T, O)           # (K, T, O) log-rate modulation

    # Per-neuron target mean spike-count/bin (lognormal, matches MC_Maze).
    base_count = np.exp(rng.normal(BASE_LOG_MEAN, BASE_LOG_STD, size=O))   # (O,)

    # Multiplicative modulation normalised so each neuron's time-mean is 1; this
    # makes GAIN control modulation depth while preserving the target mean rate.
    mod = np.exp(GAIN * z)
    mod = mod / mod.mean(axis=(0, 1), keepdims=True)

    rate = base_count[None, None, :] * mod       # (K, T, O) Poisson mean per bin
    spikes = rng.poisson(rate).astype(np.float32)
    return spikes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="synth_spikes_8planes.npy")
    args = parser.parse_args()

    data = generate(args.seed)
    np.save(args.out, data)

    nz = (data == 0).mean()
    print(f"Saved {data.shape} → {args.out}")
    print(f"  values: min={data.min():.0f} max={data.max():.0f} "
          f"mean={data.mean():.4f} counts/bin")
    print(f"  sparsity: {nz:.3%} zeros  |  ~{data.mean()/0.01:.2f} Hz mean rate")

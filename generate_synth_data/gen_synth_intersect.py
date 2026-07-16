"""
Synthetic spikes from intersecting sine waves.

Idea
----
Every channel owns TWO sine waves, each with its own frequency and phase.
Wherever a channel's two waves cross (intersect), it emits a spike.

    spike(o, t) = 1   if   wave1_o crosses wave2_o between t-1 and t

Because the two waves have different frequencies, they keep drifting in and out
of phase, so the crossings land at different, structured times for each channel.
A per-trial phase shift makes the K trials differ while keeping each channel's
frequency identity fixed.

Output: (K, T, O) binary spike array → saved as synth_intersect.npy
"""

import numpy as np

K = 200    # trials
T = 90     # timesteps
O = 137    # channels


def generate(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)                          # (T,) time in [0, 1]

    # each channel gets two waves: different frequencies and different phases
    f1 = rng.uniform(1.0, 8.0, O)
    f2 = rng.uniform(1.0, 8.0, O)
    p1 = rng.uniform(0.0, 2 * np.pi, O)
    p2 = rng.uniform(0.0, 2 * np.pi, O)

    # per-trial phase shift so each trial's crossings land in different places
    shift = rng.uniform(0.0, 2 * np.pi, (K, 1, O))

    tt = t[None, :, None]                                  # (1, T, 1)
    wave1 = np.sin(2 * np.pi * f1 * tt + p1 + shift)       # (K, T, O)
    wave2 = np.sin(2 * np.pi * f2 * tt + p2)               # (1, T, O) -> broadcasts

    diff = wave1 - wave2
    crossed = np.sign(diff)[:, 1:] != np.sign(diff)[:, :-1]   # (K, T-1, O) sign flip = crossing

    spikes = np.zeros((K, T, O), dtype=np.float32)
    spikes[:, 1:][crossed] = 1.0
    return spikes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="synth_intersect.npy")
    args = parser.parse_args()

    data = generate(args.seed)
    np.save(args.out, data)

    print(f"Saved {data.shape} → {args.out}")
    print(f"  binary values: {np.unique(data).tolist()}")
    print(f"  sparsity: {(data == 0).mean():.3%} zeros  |  "
          f"mean {data.mean():.4f} spikes/bin  |  "
          f"{data.sum(1).mean():.2f} spikes/trial/channel")

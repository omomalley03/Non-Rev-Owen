"""Standalone explainer figure for the non-reversibility metric (ζ).

Builds one composite figure contrasting a non-reversible process (a rotation /
loop) with a reversible one (a 1-D line — the ζ=0 failure case the model
sometimes produces).  Two bands:

  Band A — intuition + mechanism : forward trajectory · time-reversed
           trajectory · dim0/dim1 phase plot.
  Band B — second-order structure: space-time covariance C · reversible part
           C⁺ = C+σ(C) · non-reversible part C⁻ = C−σ(C),  with ζ = ‖C⁻‖²/‖C⁺‖².

Self-contained: no NWB load.  ζ is computed with the project's own metric
(loss.S_ratio) so the numbers on the figure are the real thing.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

from gen_synth_rosinski import get_rotations, K, T
from loss import S_ratio, _batch_rms_normalize

CMAP = "coolwarm"


# --------------------------------------------------------------------------- #
# Synthetic processes
# --------------------------------------------------------------------------- #
def make_rotation(rng) -> np.ndarray:
    """Clean 2-D rotations (cos/sin, per-trial radius+phase), shape (K,2,T)."""
    return get_rotations(rng)


def make_line(rng, alpha: float = 0.8) -> np.ndarray:
    """1-D line: both dims share the same time course → in-phase, ζ≈0.

    x0(t) sweeps through the origin (blue→red); x1 = alpha·x0.  Shape (K,2,T).
    """
    f = np.linspace(-1.0, 1.0, T)                 # one-way sweep through origin
    amp = rng.uniform(0.3, 1.0, size=K)           # per-trial amplitude
    x0 = amp[:, None] * f[None, :]                 # (K, T)
    x1 = alpha * x0
    return np.stack([x0, x1], axis=1).astype(np.float32)   # (K, 2, T)


def prep(F_np: np.ndarray):
    """Center across trials, RMS-normalise; return torch (K,2,T) and real ζ."""
    F = torch.from_numpy(F_np)
    F = F - F.mean(dim=0, keepdim=True)            # zero-mean across trials
    F = _batch_rms_normalize(F)
    zeta = S_ratio(F).item()
    return F, zeta


# --------------------------------------------------------------------------- #
# Space-time covariance  C = mean_k vec(F_kᵀ) vec(F_kᵀ)ᵀ   (2T × 2T)
# --------------------------------------------------------------------------- #
def spacetime_cov(F: torch.Tensor):
    """Return (C, C_plus, C_minus, zeta_direct).  F is (K, 2, T) torch."""
    Fn = F.numpy()
    Kk, d, Tt = Fn.shape
    V = Fn.reshape(Kk, d * Tt)                     # row k = [dim0(t), dim1(t)]
    C = (V.T @ V) / Kk                             # (2T, 2T)

    sigma = np.empty_like(C)                       # transpose each T×T block
    for i in range(d):
        for j in range(d):
            sigma[i * Tt:(i + 1) * Tt, j * Tt:(j + 1) * Tt] = \
                C[i * Tt:(i + 1) * Tt, j * Tt:(j + 1) * Tt].T

    C_plus = C + sigma
    C_minus = C - sigma
    zeta_direct = (C_minus ** 2).sum() / (C_plus ** 2).sum()
    return C, C_plus, C_minus, float(zeta_direct)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def plot_traj(ax, F, reverse=False, n_show=5):
    Fn = F.numpy()
    if reverse:
        Fn = Fn[:, :, ::-1]
    cmap = plt.get_cmap(CMAP)
    Tt = Fn.shape[2]
    idx = np.linspace(0, Fn.shape[0] - 1, n_show).astype(int)
    for k in idx:
        x, y = Fn[k, 0], Fn[k, 1]
        for t in range(Tt - 1):
            ax.plot(x[t:t + 2], y[t:t + 2], color=cmap(t / (Tt - 1)), lw=1.3)
        ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-4], y[-4]),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(1.0), lw=1.4))
    ax.axhline(0, color="k", lw=0.4, alpha=0.25)
    ax.axvline(0, color="k", lw=0.4, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])


def plot_phase(ax, F):
    Fn = F.numpy()
    energy = (Fn ** 2).sum(axis=(1, 2))
    k = int(np.argsort(energy)[len(energy) // 2])   # median-energy trial
    t = np.arange(Fn.shape[2])
    ax.plot(t, Fn[k, 0], color="steelblue", lw=2, label="dim 0")
    ax.plot(t, Fn[k, 1], color="tomato", lw=2, label="dim 1")
    ax.axhline(0, color="k", lw=0.4, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("time", fontsize=8)
    ax.set_yticks([])
    ax.legend(fontsize=7, frameon=False, loc="upper right")


def plot_cov(ax, M, title, Tt, vmax=None):
    v = np.abs(M).max() if vmax is None else vmax
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.axhline(Tt - 0.5, color="k", lw=0.6)
    ax.axvline(Tt - 0.5, color="k", lw=0.6)
    ax.set_xticks([Tt / 2, Tt + Tt / 2]); ax.set_xticklabels(["dim 0", "dim 1"], fontsize=7)
    ax.set_yticks([Tt / 2, Tt + Tt / 2]); ax.set_yticklabels(["dim 0", "dim 1"], fontsize=7,
                                                             rotation=90, va="center")
    ax.set_title(title, fontsize=9)
    return im


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(0)

    Frot, zr = prep(make_rotation(rng))
    Flin, zl = prep(make_line(rng))

    Cr, Cpr, Cmr, zr_d = spacetime_cov(Frot)
    Cl, Cpl, Cml, zl_d = spacetime_cov(Flin)

    print(f"rotation : ζ(S_ratio) = {zr:.3f}   ζ(direct ‖C⁻‖²/‖C⁺‖²) = {zr_d:.3f}")
    print(f"line     : ζ(S_ratio) = {zl:.3f}   ζ(direct ‖C⁻‖²/‖C⁺‖²) = {zl_d:.3f}")

    fig = plt.figure(figsize=(12, 13), constrained_layout=True)
    fig.suptitle("What makes a trajectory non-reversible?", fontsize=15, fontweight="bold")

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0],
                              hspace=0.08)

    # ---- Band A: intuition + mechanism (rotation row, line row) ----
    gsA = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0], wspace=0.25, hspace=0.3)
    band_a = [
        ("Non-reversible: rotation", Frot, zr),
        ("Reversible: 1-D line",     Flin, zl),
    ]
    for r, (label, F, z) in enumerate(band_a):
        ax_f = fig.add_subplot(gsA[r, 0])
        plot_traj(ax_f, F, reverse=False)
        ax_f.set_title(f"Forward     ζ = {z:.2f}", fontsize=9)
        ax_f.set_ylabel(label, fontsize=10, fontweight="bold")

        ax_r = fig.add_subplot(gsA[r, 1])
        plot_traj(ax_r, F, reverse=True)
        verdict = "different ⇒ ζ→1" if r == 0 else "identical ⇒ ζ→0"
        ax_r.set_title(f"Time-reversed   ({verdict})", fontsize=9)

        ax_p = fig.add_subplot(gsA[r, 2])
        plot_phase(ax_p, F)
        phase_note = "90° lead — out of phase ⇒ loop" if r == 0 else "in phase ⇒ line"
        ax_p.set_title(phase_note, fontsize=9)

    # ---- Band B: second-order structure ----
    gsB = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], wspace=0.3, hspace=0.35)
    band_b = [
        ("rotation", Cr, Cpr, Cmr, zr_d),
        ("line",     Cl, Cpl, Cml, zl_d),
    ]
    # shared color scale across all of Band B, so the line's C⁻ reads as flat
    # white (≈0) next to the rotation's strong bands instead of autoscaled noise.
    gvmax = max(np.abs(M).max() for _, C, Cp, Cm, _ in band_b for M in (C, Cp, Cm))
    cov_axes, last_im = [], None
    for r, (name, C, Cp, Cm, zd) in enumerate(band_b):
        a0 = fig.add_subplot(gsB[r, 0]); plot_cov(a0, C,  "C  (space-time cov.)", T, gvmax)
        a0.set_ylabel(name, fontsize=10, fontweight="bold")
        a1 = fig.add_subplot(gsB[r, 1]); plot_cov(a1, Cp, "C⁺ = C+σ(C)   reversible", T, gvmax)
        a2 = fig.add_subplot(gsB[r, 2]); last_im = plot_cov(a2, Cm, "C⁻ = C−σ(C)   non-reversible", T, gvmax)
        a2.text(1.04, 0.5, f"ζ = ‖C⁻‖²/‖C⁺‖²\n   = {zd:.2f}",
                transform=a2.transAxes, fontsize=10, va="center", ha="left")
        cov_axes += [a0, a1, a2]

    fig.colorbar(last_im, ax=cov_axes, fraction=0.012, pad=0.02,
                 label="covariance  (red + / blue − / white 0)")

    fig.text(0.5, -0.01,
             "Non-reversibility = a consistent phase lead between dimensions (a loop). "
             "Reverse time → the lead flips sign (C⁻ → −C⁻), |ζ| unchanged. "
             "No lead (a line) → C⁻ = 0 → ζ = 0.",
             ha="center", fontsize=10, style="italic")

    fig.savefig("fig_nonrev_explainer.png", dpi=200, bbox_inches="tight")
    fig.savefig("fig_nonrev_explainer.svg", bbox_inches="tight")
    print("saved fig_nonrev_explainer.png / .svg")
    plt.show()


if __name__ == "__main__":
    main()

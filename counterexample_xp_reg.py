"""
Counter-examples showing the cross-plane non-rev regularizer
measures the wrong thing.

Example 1 (false positive): HIGH regularizer, but planes are INDEPENDENT.
  Two independent rotations at the same frequency. Random shuffles pair
  sin/cos from different planes → high S.

Example 2 (false negative): LOW regularizer, but planes are COLLAPSED.
  Plane 1 is a 90-degree rotation of plane 0 in dim-space: (dim2,dim3) =
  (dim1, dim0). The planes span the exact same subspace (identical info),
  but the cross-plane regularizer doesn't detect it because cross-mixing
  produces (cos,cos) or (sin,sin) pairs that don't rotate.

Run:  python counterexample_xp_reg.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from loss import non_reversibility_S, _batch_rms_normalize, non_rev_regularizer


def corr_matrix(F):
    """(K, d, T) -> (d, d) Pearson correlation matrix."""
    K, d, T = F.shape
    Z = F.permute(0, 2, 1).reshape(K * T, d)
    Z = Z - Z.mean(dim=0)
    Z = Z / (Z.std(dim=0) + 1e-8)
    return ((Z.T @ Z) / Z.shape[0]).numpy()


def plot_example(title, F, reg_mean, reg_std, corr_off_diag, ax_row):
    """Plot rotations for one example across a row of axes.

    ax_row: list of 5 axes [plane0, plane1, cross(0,2), cross(1,3), corr_mat]
    """
    K, d, T = F.shape
    F_np = F.numpy()
    n_show = min(12, K)

    trial_colors = cm.tab10(np.linspace(0, 1, n_show))
    time_colors = cm.viridis(np.linspace(0, 1, T))

    pairs = [
        (0, 1, "dim 0", "dim 1"),   # plane 0
        (2, 3, "dim 2", "dim 3"),   # plane 1
        (0, 2, "dim 0", "dim 2"),   # cross A
        (1, 3, "dim 1", "dim 3"),   # cross B
    ]

    for col, (dx, dy, xl, yl) in enumerate(pairs):
        ax = ax_row[col]
        for k in range(n_show):
            x, y = F_np[k, dx, :], F_np[k, dy, :]
            # time-coded color per segment
            for ti in range(T - 1):
                ax.plot(x[ti:ti+2], y[ti:ti+2], color=time_colors[ti],
                        alpha=0.6, lw=1.0)
            # start marker
            ax.scatter(x[0], y[0], s=25, color='black', zorder=5, marker='o')
            # end marker
            ax.scatter(x[-1], y[-1], s=25, color='red', zorder=5, marker='s')
        ax.set_xlabel(xl, fontsize=8)
        if col == 0:
            ax.set_ylabel(yl, fontsize=8)
        else:
            ax.set_ylabel(yl, fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        # light grey grid
        ax.grid(True, alpha=0.2, lw=0.5)

    # draw vertical separator between plane plots and cross-pair plots
    for col in [2, 3]:
        ax_row[col].spines['left'].set_color('#cc0000')
        ax_row[col].spines['left'].set_linewidth(2)

    # Panel 5: Correlation matrix heatmap
    ax = ax_row[4]
    C = corr_matrix(F)
    im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f'd{i}' for i in range(4)], fontsize=7)
    ax.set_yticklabels([f'd{i}' for i in range(4)], fontsize=7)
    # plane-block outlines
    for p in range(2):
        rect = plt.Rectangle((2*p - 0.5, 2*p - 0.5), 2, 2, fill=False,
                              edgecolor='black', lw=1.5, linestyle='--')
        ax.add_patch(rect)
    # annotate cells
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{C[i,j]:+.2f}", ha='center', va='center', fontsize=6,
                    color='white' if abs(C[i,j]) > 0.6 else 'black')

    # Reg value annotation below the cross-pair plots
    ax_row[2].annotate(f"Reg = {reg_mean:.2f} ± {reg_std:.2f}",
                       xy=(0.5, -0.15), xycoords='axes fraction',
                       fontsize=8, ha='center', color='#cc0000', fontweight='bold')


def report(name, F):
    """Print non-rev scores and correlation matrix for a 4d embedding."""
    F_hat = _batch_rms_normalize(F)
    S_full = non_reversibility_S(F_hat).item()
    S_p0 = non_reversibility_S(F_hat[:, 0:2, :]).item()
    S_p1 = non_reversibility_S(F_hat[:, 2:4, :]).item()

    # targeted swap: pair (dim0,dim2) and (dim1,dim3)
    idx_cross = [2, 0, 3, 1]
    S_cross = non_reversibility_S(F_hat[:, idx_cross, :]).item()

    reg_vals = [non_rev_regularizer(F_hat).item() for _ in range(500)]
    S_reg_mean = np.mean(reg_vals)
    S_reg_std = np.std(reg_vals)

    C = corr_matrix(F)
    off_diag = np.abs(C[0:2, 2:4]).mean()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Within-plane S:   plane 0 = {S_p0:.4f}   plane 1 = {S_p1:.4f}")
    print(f"  Full S (both):    {S_full:.4f}")
    print(f"  Cross-plane S:    {S_cross:.4f}   (targeted swap)")
    print(f"  Regularizer:      {S_reg_mean:.4f} +/- {S_reg_std:.4f}  (500 shuffles)")
    print(f"  Cross-plane |r|:  {off_diag:.4f}")
    print(f"\n  Correlation matrix:")
    for i in range(4):
        row = "    " + "  ".join(f"{C[i,j]:+.3f}" for j in range(4))
        print(row)
    print()

    return S_reg_mean, S_reg_std, off_diag


K, T = 256, 64
torch.manual_seed(42)
t = torch.linspace(0, 2 * np.pi, T).unsqueeze(0)          # (1, T)

# ── Example 1: FALSE POSITIVE ──────────────────────────────────────────
# Two independent rotations at the SAME frequency.
# They carry different info (independent phases) but the regularizer
# penalises them because shuffled dims form cross-plane rotation pairs.

phi0 = torch.randn(K, 1)
phi1 = torch.randn(K, 1)        # independent of phi0
plane0 = torch.stack([torch.cos(t + phi0), torch.sin(t + phi0)], dim=1)
plane1 = torch.stack([torch.cos(t + phi1), torch.sin(t + phi1)], dim=1)
F_indep = torch.cat([plane0, plane1], dim=1)

r1 = report("Ex 1 — FALSE POSITIVE\n"
             "  Independent planes, same frequency\n"
             "  Regularizer HIGH despite planes being non-redundant", F_indep)

# ── Example 2a: FALSE NEGATIVE — dim-swapped collapse ─────────────────
# Plane 1 is plane 0 with dims swapped: (dim2, dim3) = (dim1, dim0).
# Planes span the EXACT same 2D subspace — total redundancy.
# But the regularizer value ≈ Ex 3 value, so it can't distinguish this
# from independent planes. More critically: the *targeted* swap gives
# high S, but random shuffles include pairings that break rotation too.

phi = torch.randn(K, 1)
plane0 = torch.stack([torch.cos(t + phi), torch.sin(t + phi)], dim=1)
plane1 = torch.stack([torch.sin(t + phi), torch.cos(t + phi)], dim=1)  # swapped
F_swapped = torch.cat([plane0, plane1], dim=1)

r2a = report("Ex 2a — FALSE NEGATIVE (dim-swapped collapse)\n"
             "  Planes span identical subspace\n"
             "  Reg ≈ same as independent planes (can't distinguish)", F_swapped)

# ── Example 2b: FALSE NEGATIVE — non-linear redundancy ────────────────
# Two planes at different frequencies sharing an amplitude envelope.
# dim-level |r| ≈ 0 (orthogonal waveforms), but mutual information is
# high (shared modulation). Neither the regularizer NOR Barlow Twins
# can detect this kind of redundancy.

phi = torch.randn(K, 1)
amp = (1.0 + 0.8 * torch.sin(0.5 * t + phi))       # shared envelope
plane0 = torch.stack([amp * torch.cos(t + phi),
                      amp * torch.sin(t + phi)], dim=1)
plane1 = torch.stack([amp * torch.cos(3*t + phi),
                      amp * torch.sin(3*t + phi)], dim=1)
F_shared_amp = torch.cat([plane0, plane1], dim=1)

r2b = report("Ex 2b — FALSE NEGATIVE (non-linear redundancy)\n"
             "  Shared amplitude, different frequencies\n"
             "  Both regularizer AND Barlow Twins miss this", F_shared_amp)

# ── Example 3: for comparison — TRUE COLLAPSE that IS detected ────────
# Plane 1 = scaled copy of plane 0 (same dim alignment).
# Here the regularizer DOES work because cross-mixing (cos, scaled-sin)
# still rotates. Shows the regularizer only catches aligned-copy collapse.

plane0 = torch.stack([torch.cos(t + phi), torch.sin(t + phi)], dim=1)
plane1 = 0.8 * plane0 + 0.05 * torch.randn_like(plane0)
F_copy = torch.cat([plane0, plane1], dim=1)

r3 = report("Ex 3 — TRUE POSITIVE (for comparison)\n"
            "  Plane 1 = 0.8 * Plane 0 (aligned copy)\n"
            "  Regularizer catches this type of collapse", F_copy)

# ── Figure ─────────────────────────────────────────────────────────────
examples = [
    ("Ex 1: FALSE POSITIVE\nIndependent planes, same freq\n(NOT collapsed)", F_indep, r1),
    ("Ex 2a: FALSE NEGATIVE\nDim-swapped identical subspace\n(TOTAL collapse)", F_swapped, r2a),
    ("Ex 2b: FALSE NEGATIVE\nShared amplitude, diff freq\n(Nonlinear redundancy)", F_shared_amp, r2b),
    ("Ex 3: TRUE POSITIVE\nAligned copy (0.8x)\n(Linear collapse)", F_copy, r3),
]

fig, axes = plt.subplots(4, 5, figsize=(20, 16),
                         gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.6],
                                      'wspace': 0.35, 'hspace': 0.55})

col_headers = ["Plane 0\n(dim0 vs dim1)",
               "Plane 1\n(dim2 vs dim3)",
               "Cross-pair A\n(dim0 vs dim2)",
               "Cross-pair B\n(dim1 vs dim3)",
               "Correlation\nmatrix"]

row_labels = [
    "Ex 1: FALSE POSITIVE\nIndependent planes\nReg = {reg:.2f}",
    "Ex 2a: FALSE NEGATIVE\nDim-swapped collapse\nReg = {reg:.2f}",
    "Ex 2b: FALSE NEGATIVE\nShared-amplitude\nReg = {reg:.2f}",
    "Ex 3: TRUE POSITIVE\nAligned copy (0.8x)\nReg = {reg:.2f}",
]

for row_idx, (title, F, (reg_mean, reg_std, off_diag)) in enumerate(examples):
    plot_example(title, F, reg_mean, reg_std, off_diag, axes[row_idx])

# Column headers on top row only
for col, label in enumerate(col_headers):
    axes[0, col].set_title(label, fontsize=10, fontweight='bold', pad=12)

# Row labels on the left
row_colors = ['#2ca02c', '#d62728', '#d62728', '#1f77b4']
for row_idx, (_, _, (reg_mean, reg_std, off_diag)) in enumerate(examples):
    label = row_labels[row_idx].format(reg=reg_mean)
    fig.text(0.01, 0.87 - row_idx * 0.23, label,
             fontsize=10, fontweight='bold', va='center', ha='left',
             color=row_colors[row_idx],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=row_colors[row_idx], alpha=0.9))

# Bracket between col 1 and col 2: "within-plane" vs "cross-plane"
fig.text(0.27, 0.96, "Within-plane rotations",
         fontsize=10, ha='center', style='italic', color='#333333')
fig.text(0.59, 0.96, "Cross-plane pairings (what regularizer sees)",
         fontsize=10, ha='center', style='italic', color='#cc0000')

# Time color bar legend
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.08, 0.01, 0.12])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('time (0→2π)', fontsize=8)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['start', 'end'], fontsize=7)

# marker legend
fig.text(0.92, 0.22, '● start   ■ end', fontsize=8, ha='left', color='#333')

fig.suptitle("Cross-Plane Non-Rev Regularizer: Counter-Examples",
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig("counterexample_xp_reg.png", dpi=150, bbox_inches='tight',
            facecolor='white')
print(f"\nFigure saved to: counterexample_xp_reg.png")
plt.close()

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  {'Example':<45} {'Reg':>7}  {'|r|':>5}  {'Collapsed?'}")
print(f"  {'-------':<45} {'---':>7}  {'---':>5}  {'----------'}")
print(f"  {'Ex 1: Independent planes (same freq)':<45} {r1[0]:>7.2f}  {r1[2]:>5.2f}  {'NO'}")
print(f"  {'Ex 2a: Dim-swapped identical subspace':<45} {r2a[0]:>7.2f}  {r2a[2]:>5.2f}  {'YES (total)'}")
print(f"  {'Ex 2b: Shared amplitude, diff freq':<45} {r2b[0]:>7.2f}  {r2b[2]:>5.2f}  {'YES (nonlinear)'}")
print(f"  {'Ex 3: Aligned copy (0.8x)':<45} {r3[0]:>7.2f}  {r3[2]:>5.2f}  {'YES (linear)'}")
print()
print("  Problems with the cross-plane non-rev regularizer:")
print("  1. FALSE POSITIVE: penalises Ex 1 (independent, non-collapsed)")
print("  2. INDISTINGUISHABLE: Ex 2a, 2b, 3 all give reg ~ 1.2-1.3")
print("     despite different collapse types (and one isn't collapsed!)")
print("  3. ENORMOUS VARIANCE: std ~ 0.9 on most examples -> noisy gradients")
print("  4. BLIND TO NON-LINEAR REDUNDANCY: Ex 2b has high mutual info")
print("     but near-zero linear correlation -- invisible to all linear methods")
print("="*60)

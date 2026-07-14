import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Data
# -----------------------
d = np.array([32, 64, 128, 256, 512, 1024, 2048])

hidden_rmse = np.array([78.87, 80.27, 77.92, 79.91, 79.15, 79.28, 79.43])
hidden_r2   = np.array([0.892, 0.888, 0.895, 0.889, 0.891, 0.891, 0.890])

output_rmse = np.array([89.43, 85.57, 85.57, 85.67, 87.20, 85.89, 83.24])
output_r2   = np.array([0.862, 0.873, 0.873, 0.873, 0.869, 0.872, 0.880])

# -----------------------
# Plot
# -----------------------
fig, ax1 = plt.subplots(figsize=(7.5, 5))

# Left axis: RMSE
l1 = ax1.plot(
    d, hidden_rmse,
    marker='o', linewidth=2, markersize=6,
    label='Hidden RMSE'
)
l2 = ax1.plot(
    d, output_rmse,
    marker='s', linewidth=2, markersize=6,
    label='Output RMSE'
)

ax1.set_xscale('log', base=2)
ax1.set_xticks(d)
ax1.set_xticklabels([str(x) for x in d])
ax1.set_xlabel("Embedding dimension (log scale)")
ax1.set_ylabel("Decoder RMSE [mm/s]")
ax1.grid(True, alpha=0.3)

# Right axis: R²
ax2 = ax1.twinx()

l3 = ax2.plot(
    d, hidden_r2,
    marker='^', linewidth=2, markersize=6,
    linestyle='--',
    label='Hidden $R^2$'
)
l4 = ax2.plot(
    d, output_r2,
    marker='D', linewidth=2, markersize=6,
    linestyle='--',
    label='Output $R^2$'
)

ax2.set_ylabel(r"$R^2$")
ax2.set_ylim(0.85, 0.90)

# Combine legends
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right', frameon=True)

plt.title("MC Maze nonlinear multiscale embeddings: decoder performance")
plt.tight_layout()
plt.tight_layout()

plt.savefig(
    "decoder_performance_vs_embedding_dimension.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "decoder_performance_vs_embedding_dimension.png",
    dpi=300,
    bbox_inches="tight"
)
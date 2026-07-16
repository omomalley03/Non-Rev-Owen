import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Data
# -----------------------
d = np.array([32, 64, 128, 256, 512, 1024, 2048])

hidden_r2 = np.array([0.892, 0.888, 0.895, 0.889, 0.891, 0.891, 0.890])
output_r2 = np.array([0.862, 0.873, 0.873, 0.873, 0.869, 0.872, 0.880])

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(7, 5))

plt.plot(
    d, hidden_r2,
    marker='o',
    linewidth=2,
    markersize=6,
    label='Hidden layer'
)

plt.plot(
    d, output_r2,
    marker='s',
    linewidth=2,
    markersize=6,
    label='Output layer'
)

plt.xscale('log', base=2)
plt.xticks(d, [str(x) for x in d])

plt.xlabel("Embedding dimension (log scale)")
plt.ylabel(r"$R^2$")
plt.title(r"Hand velocity prediction $R^2$ vs. embedding dimension")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "r2_vs_embedding_dimension.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "r2_vs_embedding_dimension.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
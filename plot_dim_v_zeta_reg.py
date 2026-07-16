import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Data
# -----------------------
d = np.array([32, 64, 128, 256, 512, 1024, 2048])

zeta = np.array([0.44, 0.42, 0.455, 0.428, 0.413, 0.417, 0.414])
reg  = np.array([0.069, 0.145, 0.149, 0.168, 0.174, 0.176, 0.176])

# -----------------------
# Plot
# -----------------------
fig, ax1 = plt.subplots(figsize=(7, 5))

# Left axis: zeta
l1 = ax1.plot(
    d, zeta,
    color='green',
    marker='o',
    linewidth=2,
    markersize=6,
    label=r'$\zeta$'
)

ax1.set_xscale('log', base=2)
ax1.set_xticks(d)
ax1.set_xticklabels([str(x) for x in d])

ax1.set_xlabel("Embedding dimension (log scale)")
ax1.set_ylabel(r"Validation $\zeta$")
ax1.grid(True, alpha=0.3)

# Right axis: regularisation
ax2 = ax1.twinx()



l2 = ax2.plot(
    d, reg,
    color='red',
    marker='s',
    linewidth=2,
    markersize=6,
    linestyle='--',
    label='Regularisation'
)

ax2.set_ylabel("Regularisation loss")

# Combine legends
lines = l1 + l2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="center right")

plt.title(r"$\zeta$ and regularisation vs. embedding dimension")

plt.tight_layout()

plt.savefig(
    "zeta_reg_v_embedding_dim_mcmaze.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "zeta_reg_v_embedding_dim_mcmaze.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Data
dims = np.array([32, 64, 128, 256, 512, 1024, 2048])

zeta = np.array([0.4394, 0.4243, 0.4551, 0.4283, 0.4127, 0.4167, 0.4137])
hidden_rmse = np.array([78.87, 80.27, 77.92, 79.91, 79.15, 79.28, 79.43])
hidden_r2 = np.array([0.892, 0.888, 0.895, 0.889, 0.891, 0.891, 0.890])

# -----------------------
# Hidden RMSE vs Zeta
# -----------------------
plt.figure(figsize=(6,5))

plt.scatter(zeta, hidden_rmse, s=70)

for x, y, d in zip(zeta, hidden_rmse, dims):
    plt.annotate(str(d), (x, y), xytext=(4,4),
                 textcoords="offset points", fontsize=9)

plt.xlabel(r'Best validation $\zeta$')
plt.ylabel('Hidden decoder RMSE [mm/s]')
plt.title(r'Hidden RMSE vs. validation $\zeta$')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hidden_rmse_vs_zeta.pdf", dpi=300, bbox_inches="tight")
plt.savefig("hidden_rmse_vs_zeta.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------
# Hidden R² vs Zeta
# -----------------------
plt.figure(figsize=(6,5))

plt.scatter(zeta, hidden_r2, s=70)

for x, y, d in zip(zeta, hidden_r2, dims):
    plt.annotate(str(d), (x, y), xytext=(4,4),
                 textcoords="offset points", fontsize=9)

plt.xlabel(r'Best validation $\zeta$')
plt.ylabel(r'Hidden decoder $R^2$')
plt.title(r'Hidden $R^2$ vs. validation $\zeta$')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hidden_r2_vs_zeta.pdf", dpi=300, bbox_inches="tight")
plt.savefig("hidden_r2_vs_zeta.png", dpi=300, bbox_inches="tight")
plt.show()
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
ZETA_LABEL = r"Mean validation plane $\zeta$"


def load_ci95_metric(path, metric, selectors, x_field, x_values, fallback_mean, fallback_ci):
    """Load mean/CI rows from a sweep results_ci95.csv, with hardcoded fallback."""
    path = Path(path)
    if not path.is_file():
        return fallback_mean, fallback_ci

    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("metric") != metric:
                continue
            matched = True
            for key, expected in selectors.items():
                try:
                    matched = float(row.get(key, "nan")) == float(expected)
                except ValueError:
                    matched = row.get(key) == str(expected)
                if not matched:
                    break
            if matched:
                rows[str(row.get(x_field))] = row

    means = []
    cis = []
    for value in x_values:
        row = rows.get(str(value))
        if row is None:
            return fallback_mean, fallback_ci
        means.append(float(row["mean"]))
        cis.append(float(row["ci95_half_width"]))
    return np.array(means), np.array(cis)


# ============================================================
# Helper plotting function
# ============================================================

def make_experiment_figure(
    x_labels,
    zeta, zeta_ci,
    reg, reg_ci,
    r2, r2_ci,
    rmse, rmse_ci,
    fig_title,
    output_path=None,
    xlabel=""
):
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    metrics = [
        (zeta, zeta_ci, ZETA_LABEL),
        (reg, reg_ci, "Regularisation"),
        (r2, r2_ci, r"$R^2$"),
        (rmse, rmse_ci, "RMSE"),
    ]

    for ax, (y, yerr, ylabel) in zip(axes, metrics):
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            capsize=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle(fig_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# Experiment 1: changing lambda schedule at d = 128
# ============================================================

lambda_labels = [
    r"$\lambda=1$",
    r"$\lambda=0.4\to2.0$",
    r"$\lambda=1.0\to5.0$",
    r"$\lambda=1.0\to10.0$",
    r"$\lambda=1.0\to20.0$",
    r"$\lambda=1.0\to50.0$",
]

zeta_1 = np.array([0.4624406775, 0.4580329736, 0.4609809372, 0.4683750431, 0.4680859056, 0.4215861281])
zeta_1_ci = np.array([0.00705835126, 0.02783678933, 0.01261287577, 0.01701729976, 0.03138133188, 0.04273838809])
zeta_1, zeta_1_ci = load_ci95_metric(
    REPO_ROOT / "mcmaze" / "cca_warmup_finetune_sweep" / "results_ci95.csv",
    "mean_val_zeta",
    {"dimension": 128},
    "lambda_block_cca",
    ["1.0", "2.0", "5.0", "10.0", "20.0", "50.0"],
    zeta_1,
    zeta_1_ci,
)

reg_1 = np.array([0.1018466, 0.1581754, 0.0355628, 0.015403, 0.004732, 0.0043064])
reg_1_ci = np.array([0.00818359503, 0.00701513499, 0.01038043602, 0.003656451396, 0.002522070988, 0.001279609784])

r2_1 = np.array([0.8961347699, 0.8915804029, 0.8979864359, 0.8984918118, 0.8977631927, 0.8888941765])
r2_1_ci = np.array([0.001680032071, 0.002142026449, 0.002403301579, 0.002766026708, 0.001729360165, 0.006582882634])

rmse_1 = np.array([77.3362854, 78.96047058, 76.6128, 76.4341, 76.7371, 80.0085])
rmse_1_ci = np.array([0.2887808669, 1.155320231, 0.5981, 0.8116, 0.8531, 2.9468])

make_experiment_figure(
    x_labels=lambda_labels,
    zeta=zeta_1,
    zeta_ci=zeta_1_ci,
    reg=reg_1,
    reg_ci=reg_1_ci,
    r2=r2_1,
    r2_ci=r2_1_ci,
    rmse=rmse_1,
    rmse_ci=rmse_1_ci,
    fig_title=r"Experiment 1: Effect of Regularisation Schedule on Mean Validation Plane $\zeta$ ($d=128$)",
    output_path="experiment1_lambda_schedule.png",
    xlabel="Lambda schedule",
)


# ============================================================
# Experiment 2: changing embedding dimension for lambda 1 -> 10
# ============================================================

dim_labels = ["32", "64", "128", "256"]

zeta_2 = np.array([0.4417427149, 0.4908451868, 0.4683750431, 0.4533131593])
zeta_2_ci = np.array([0.03411378404, 0.02072021353, 0.0170, 0.01829952892])
zeta_2, zeta_2_ci = load_ci95_metric(
    REPO_ROOT / "mcmaze" / "dim_train_finetune_sweep" / "results_ci95.csv",
    "mean_val_zeta",
    {"lambda_start_frac": 0.1, "lambda_block_cca": 10.0},
    "dimension",
    dim_labels,
    zeta_2,
    zeta_2_ci,
)

reg_2 = np.array([0.0076754, 0.010012, 0.015403, 0.018973])
reg_2_ci = np.array([0.003153541131, 0.004843157817, 0.003656451396, 0.003464026815])

r2_2 = np.array([0.8814297676, 0.8907823801, 0.8984918118, 0.8967314005])
r2_2_ci = np.array([0.006241007342, 0.00697501981, 0.002766026708, 0.006506792129])

rmse_2 = np.array([82.6665, 79.3157, 76.4341, 77.0722])
rmse_2_ci = np.array([1.686, 2.7759, 0.8116, 3.0105])

make_experiment_figure(
    x_labels=dim_labels,
    zeta=zeta_2,
    zeta_ci=zeta_2_ci,
    reg=reg_2,
    reg_ci=reg_2_ci,
    r2=r2_2,
    r2_ci=r2_2_ci,
    rmse=rmse_2,
    rmse_ci=rmse_2_ci,
    fig_title=r"Experiment 2: Effect of Embedding Dimension on Mean Validation Plane $\zeta$ ($\lambda=1\to10$)",
    output_path="experiment2_embedding_dimension.png",
    xlabel="Embedding dimension",
)


# ============================================================
# Experiment 3: changing percent allocation to odd planes
# at d = 128, lambda 1 -> 10
# ============================================================

odd_plane_labels = ["0", "25", "50", "75", "100"]

zeta_3 = np.array([0.4200669361, 0.4442407191, 0.4683750431, 0.4675893399, 0.4444205834])
zeta_3_ci = np.array([0.008820295716, 0.01796734391, 0.01701729976, 0.04193343134, 0.01037684157])
zeta_3, zeta_3_ci = load_ci95_metric(
    REPO_ROOT / "mcmaze" / "odd_plane_fraction_finetune_sweep" / "results_ci95.csv",
    "mean_val_zeta",
    {"dimension": 128, "lambda_start_frac": 0.1, "lambda_block_cca": 10.0},
    "odd_plane_fraction",
    ["0.0", "0.25", "0.5", "0.75", "1.0"],
    zeta_3,
    zeta_3_ci,
)

reg_3 = np.array([0.0207278, 0.0237958, 0.015403, 0.0095126, 0.004718])
reg_3_ci = np.array([0.006693390114, 0.007686850364, 0.003656451396, 0.004494106367, 0.0006464360833])

r2_3 = np.array([0.8897231579, 0.8960544825, 0.8984918118, 0.8937672377, 0.8458788037])
r2_3_ci = np.array([0.004462398585, 0.008398103552, 0.002766026708, 0.003155249109, 0.001694410681])

rmse_3 = np.array([79.698, 77.3352, 76.4341, 78.1891, 94.1561])
rmse_3_ci = np.array([1.9227, 2.9595, 0.8116, 0.6978, 0.4402])

make_experiment_figure(
    x_labels=odd_plane_labels,
    zeta=zeta_3,
    zeta_ci=zeta_3_ci,
    reg=reg_3,
    reg_ci=reg_3_ci,
    r2=r2_3,
    r2_ci=r2_3_ci,
    rmse=rmse_3,
    rmse_ci=rmse_3_ci,
    fig_title=r"Experiment 3: Effect of Odd-Plane Allocation on Mean Validation Plane $\zeta$ ($d=128,\ \lambda=1\to10$)",
    output_path="experiment3_odd_plane_allocation.png",
    xlabel="Percent allocation to odd planes",
)

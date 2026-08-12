import numpy as np
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from scipy.stats import t


# ============================================================
# Data
# ============================================================

dimensions = np.array([16, 32, 64, 128, 256, 512])

# Shape: (number of dimensions, number of seeds)
# MC MAZE REG LAMBDA 1:
mlp_zeta_old = np.array([
    [0.382000, 0.375200, 0.412000, 0.370800, 0.373100],
    [0.356246, 0.351258, 0.359993, 0.336242, 0.349568],
    [0.326916, 0.343618, 0.326392, 0.329499, 0.356865],
    [0.332347, 0.328121, 0.320724, 0.333570, 0.324513],
    [0.320307, 0.321804, 0.319961, 0.320731, 0.324729],
    [0.326612, 0.320959, 0.322611, 0.325632, 0.327201],
])

mlp_regularisation_old = np.array([
    [0.007589, 0.008861, 0.008845, 0.016638, 0.017316],
    [0.018508, 0.032574, 0.027448, 0.028369, 0.011335],
    [0.025945, 0.031746, 0.036647, 0.038216, 0.035845],
    [0.031901, 0.032753, 0.025796, 0.040470, 0.032230],
    [0.035096, 0.030071, 0.033543, 0.033296, 0.033349],
    [0.043066, 0.044112, 0.043890, 0.042211, 0.043728],
])

# symconv_zeta = np.array([
#     [0.437511, 0.390931, 0.485046, 0.435627, 0.398432],
#     [0.502505, 0.457393, 0.457410, 0.490573, 0.476814],
#     [0.483826, 0.463242, 0.441803, 0.456186, 0.454057],
#     [0.449756, 0.419004, 0.453511, 0.460666, 0.427243],
#     [0.429890, 0.433877, 0.445639, 0.434466, 0.431145],
#     [0.433014, 0.415125, 0.422862, 0.417542, 0.425860],
# ])

# symconv_regularisation = np.array([
#     [0.015497, 0.018059, 0.022987, 0.013664, 0.020161],
#     [0.089883, 0.054795, 0.091698, 0.060835, 0.063311],
#     [0.113017, 0.121825, 0.111390, 0.105608, 0.113827],
#     [0.143513, 0.133908, 0.138473, 0.144862, 0.133806],
#     [0.156791, 0.154129, 0.155538, 0.145763, 0.161573],
#     [0.157588, 0.164016, 0.149215, 0.157244, 0.167210],
# ])

# MCMAZE LAMBDA 1 TO 5

mlp_zeta = np.array([
    [0.378301, 0.379652, 0.380570, 0.336144, 0.343170],
    [0.350658, 0.342156, 0.319028, 0.324541, 0.339235],
    [0.292242, 0.302867, 0.303160, 0.313089, 0.302083],
    [0.297800, 0.295046, 0.287101, 0.289667, 0.292355],
    [0.274712, 0.271367, 0.274632, 0.272180, 0.273492],
    [0.285733, 0.286061, 0.288795, 0.287753, 0.287968],
])

mlp_regularisation = np.array([
    [0.009144, 0.009160, 0.009365, 0.020903, 0.007539],
    [0.031083, 0.029906, 0.027808, 0.024617, 0.016466],
    [0.019799, 0.058350, 0.026774, 0.024166, 0.021282],
    [0.023116, 0.025051, 0.037471, 0.024090, 0.037913],
    [0.026096, 0.034380, 0.032456, 0.024711, 0.033633],
    [0.051971, 0.056739, 0.054266, 0.054143, 0.054442],
])


# #PHYSIONET
# mlp_zeta = np.array([
#     [0.246869, 0.257083, 0.263283, 0.255679, 0.269984],
#     [0.239073, 0.245799, 0.271818, 0.258312, 0.226947],
#     [0.238231, 0.250524, 0.261864, 0.244280, 0.236601],
#     [0.218983, 0.230376, 0.227154, 0.224270, 0.224654],
#     [0.201670, 0.236187, 0.243139, 0.227028, 0.213610],
#     [0.205130, 0.230407, 0.226141, 0.227878, 0.219021],
# ])

# mlp_regularisation = np.array([
#     [0.003547, 0.002375, 0.001815, 0.003587, 0.001729],
#     [0.002015, 0.003001, 0.002641, 0.001356, 0.002854],
#     [0.003915, 0.004603, 0.005471, 0.005470, 0.008210],
#     [0.020977, 0.005133, 0.005292, 0.017257, 0.031397],
#     [0.105429, 0.042715, 0.011149, 0.017198, 0.019390],
#     [0.021316, 0.041534, 0.022510, 0.021954, 0.020346],
# ])

# symconv_zeta = np.array([
#     [0.316027, 0.309629, 0.328904, 0.307715, 0.324130],
#     [0.306152, 0.306489, 0.287735, 0.285297, 0.302360],
#     [0.299716, 0.305820, 0.298038, 0.300845, 0.300815],
#     [0.291443, 0.268709, 0.268522, 0.280679, 0.296732],
#     [0.306742, 0.310602, 0.250926, 0.292083, 0.262233],
#     [0.269385, 0.252366, 0.253802, 0.249798, 0.270319],
# ])

# symconv_regularisation = np.array([
#     [0.000416, 0.001036, 0.001192, 0.002846, 0.001567],
#     [0.003981, 0.033754, 0.002819, 0.022203, 0.024295],
#     [0.056120, 0.024545, 0.101334, 0.011542, 0.089479],
#     [0.010488, 0.006297, 0.005452, 0.127644, 0.019767],
#     [0.069495, 0.090598, 0.007394, 0.056154, 0.121086],
#     [0.102295, 0.010626, 0.015362, 0.010940, 0.010757],
# ])
# ============================================================
# Statistics
# ============================================================

def mean_and_ci95(values):
    """
    Compute the mean and two-sided 95% Student's t confidence
    interval half-width across seeds.

    Parameters
    ----------
    values : np.ndarray
        Shape: (number of dimensions, number of seeds)

    Returns
    -------
    mean : np.ndarray
        Mean across seeds for each dimension.

    ci95 : np.ndarray
        Half-width of the 95% confidence interval.
    """
    n_seeds = values.shape[1]

    mean = np.mean(values, axis=1)
    sample_std = np.std(values, axis=1, ddof=1)
    standard_error = sample_std / np.sqrt(n_seeds)

    t_critical = t.ppf(0.975, df=n_seeds - 1)
    ci95 = t_critical * standard_error

    return mean, ci95


# ============================================================
# Plotting
# ============================================================

def plot_model_comparison(
    dimensions,
    mlp_values,
    symconv_values,
    ylabel,
    title,
    output_path,
):
    mlp_mean, mlp_ci95 = mean_and_ci95(mlp_values)
    symconv_mean, symconv_ci95 = mean_and_ci95(symconv_values)

    plt.figure(figsize=(7.5, 5.5))

    plt.errorbar(
        dimensions,
        mlp_mean,
        yerr=mlp_ci95,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=7,
        capsize=5,
        capthick=1.5,
        label=r"$\lambda$=1",
    )

    plt.errorbar(
        dimensions,
        symconv_mean,
        yerr=symconv_ci95,
        marker="s",
        linestyle="--",
        linewidth=2,
        markersize=7,
        capsize=5,
        capthick=1.5,
        label=r"$\lambda$=1$\to$5",
    )

    plt.xscale("log", base=2)
    plt.xticks(dimensions, [str(d) for d in dimensions])

    plt.xlabel("Embedding dimension")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Generate figures
# ============================================================

plot_model_comparison(
    dimensions=dimensions,
    mlp_values=mlp_zeta,
    symconv_values=mlp_zeta_old,
    ylabel=r"Validation $\zeta$",
    title=r"MLP Validation $\zeta$ vs. Embedding Dimension",
    output_path="mcmaze_zeta_vs_dimension_betterlambda.png",
)

plot_model_comparison(
    dimensions=dimensions,
    mlp_values=mlp_regularisation,
    symconv_values=mlp_regularisation_old,
    ylabel="Regularisation",
    title="MLP Regularisation vs. Embedding Dimension",
    output_path="mcmaze_regularisation_vs_dimension_betterlambda.png",
)
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Embedding dimensions
# -----------------------
d = np.array([32, 64, 128, 256, 512, 1024, 2048])


# -----------------------
# Data copied from sheet
#
# Array shape:
#   rows    = embedding dimensions
#   columns = decoder seeds 0, 1, 2, 3, 4
# -----------------------
results = {
    "Frozen": {
        "Hidden": {
            "RMSE": np.array([
                [79.2000, 79.4300, 79.5400, 79.9400, 79.1757],
                [80.5045, 80.5468, 80.5608, 80.4607, 80.9940],
                [78.3055, 78.6076, 78.9701, 78.2974, 79.2589],
                [80.4573, 80.4422, 80.0500, 80.6801, 80.2160],
                [79.3883, 79.7419, 79.8220, 79.4406, 79.5766],
                [79.8551, 78.7575, 79.2344, 79.7169, 79.7002],
                [79.8660, 79.5189, 79.6117, 79.6450, 79.6705],
            ]),
            "R2": np.array([
                [0.8913, 0.8905, 0.8904, 0.8893, 0.8913],
                [0.8874, 0.8874, 0.8876, 0.8877, 0.8864],
                [0.8940, 0.8931, 0.8920, 0.8939, 0.8914],
                [0.8877, 0.8880, 0.8891, 0.8868, 0.8886],
                [0.8907, 0.8898, 0.8896, 0.8904, 0.8901],
                [0.8894, 0.8925, 0.8911, 0.8898, 0.8899],
                [0.8895, 0.8903, 0.8899, 0.8901, 0.8901],
            ]),
        },
        "Output": {
            "RMSE": np.array([
                [81.8000, 81.7900, 81.7185, 81.9179, 81.7804],
                [81.5246, 81.3345, 81.3755, 81.1585, 81.4629],
                [79.7279, 80.1521, 79.8822, 80.3761, 79.8360],
                [80.3541, 80.9985, 80.9734, 81.3278, 81.0028],
                [81.9754, 82.3150, 81.8192, 81.6587, 81.3663],
                [82.9517, 83.4282, 82.7750, 82.9958, 83.4543],
                [84.0781, 84.0702, 84.1862, 84.0918, 84.5608],
            ]),
            "R2": np.array([
                [0.8841, 0.8841, 0.8843, 0.8837, 0.8840],
                [0.8849, 0.8854, 0.8853, 0.8859, 0.8851],
                [0.8898, 0.8888, 0.8895, 0.8882, 0.8895],
                [0.8881, 0.8862, 0.8866, 0.8853, 0.8862],
                [0.8839, 0.8826, 0.8840, 0.8845, 0.8854],
                [0.8810, 0.8798, 0.8816, 0.8807, 0.8795],
                [0.8776, 0.8777, 0.8775, 0.8776, 0.8761],
            ]),
        },
    },

    "Fine-tuned": {
        "Hidden": {
            "RMSE": np.array([
                [73.9300, 74.1700, 74.0100, 73.9100, 73.8703],
                [74.0935, 73.8414, 74.3109, 73.9659, 73.7704],
                [72.5888, 71.8106, 72.5724, 72.3172, 72.1968],
                [73.7159, 73.9425, 73.5564, 73.2289, 73.5880],
                [72.8500, 72.9818, 72.9750, 73.3601, 72.8720],
                [73.0933, 73.2337, 73.4045, 72.8668, 73.3693],
                [73.5356, 73.5676, 73.6243, 73.6690, 73.7317],
            ]),
            "R2": np.array([
                [0.9052, 0.9048, 0.9052, 0.9053, 0.9054],
                [0.9049, 0.9056, 0.9045, 0.9053, 0.9056],
                [0.9088, 0.9109, 0.9089, 0.9094, 0.9099],
                [0.9058, 0.9054, 0.9062, 0.9070, 0.9061],
                [0.9081, 0.9078, 0.9081, 0.9067, 0.9081],
                [0.9074, 0.9071, 0.9066, 0.9081, 0.9068],
                [0.9063, 0.9061, 0.9060, 0.9061, 0.9059],
            ]),
        },
        "Output": {
            "RMSE": np.array([
                [72.0700, 72.2900, 72.4502, 72.4075, 72.4048],
                [72.8027, 72.8677, 72.8668, 72.8835, 72.1313],
                [71.8313, 71.8201, 72.0228, 72.1488, 72.0472],
                [72.6935, 72.3368, 72.2230, 72.7652, 72.6736],
                [73.1435, 73.7741, 72.9881, 73.0466, 73.7386],
                [73.7921, 74.0940, 74.1722, 74.2744, 74.0346],
                [74.4948, 74.8202, 74.4938, 74.8698, 74.5290],
            ]),
            "R2": np.array([
                # The 0.8841 value below is copied exactly from the sheet.
                # It may be a typo because the other four seeds are near 0.909.
                [0.9101, 0.8841, 0.9092, 0.9092, 0.9091],
                [0.9082, 0.9081, 0.9080, 0.9080, 0.9099],
                [0.9106, 0.9107, 0.9102, 0.9099, 0.9102],
                [0.9084, 0.9095, 0.9096, 0.9082, 0.9086],
                [0.9074, 0.9058, 0.9079, 0.9077, 0.9061],
                [0.9059, 0.9050, 0.9050, 0.9046, 0.9052],
                [0.9041, 0.9032, 0.9040, 0.9030, 0.9040],
            ]),
        },
    },
}


# -----------------------
# Mean and 95% CI
# -----------------------
def mean_and_ci95(trials):
    """
    Calculate the mean and two-sided 95% Student-t confidence
    interval across the five decoder seeds.

    Parameters
    ----------
    trials : ndarray, shape (n_dimensions, n_seeds)

    Returns
    -------
    mean : ndarray
        Mean across seeds.

    ci95 : ndarray
        Half-width of the 95% confidence interval.
    """
    n_seeds = trials.shape[1]

    mean = np.mean(trials, axis=1)
    sample_std = np.std(trials, axis=1, ddof=1)
    standard_error = sample_std / np.sqrt(n_seeds)

    # t_(0.975, df=4), since there are five seeds
    t_critical = 2.776445

    ci95 = t_critical * standard_error

    return mean, ci95


# -----------------------
# Plotting function
# -----------------------
def plot_result(model, metric):
    hidden_mean, hidden_ci = mean_and_ci95(
        results[model]["Hidden"][metric]
    )
    output_mean, output_ci = mean_and_ci95(
        results[model]["Output"][metric]
    )

    if metric == "RMSE":
        ylabel = "RMSE"
        metric_title = "RMSE"
        metric_filename = "rmse"
    else:
        ylabel = r"$R^2$"
        metric_title = r"$R^2$"
        metric_filename = "r2"

    model_filename = model.lower().replace("-", "_")

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        d,
        hidden_mean,
        yerr=hidden_ci,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        label="Hidden layer",
    )

    plt.errorbar(
        d,
        output_mean,
        yerr=output_ci,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=6,
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        label="Output layer",
    )

    plt.xscale("log", base=2)
    plt.xticks(d, [str(value) for value in d])

    plt.xlabel("Embedding dimension (log scale)")
    plt.ylabel(ylabel)
    plt.title(
        f"Hand velocity prediction {metric_title}: "
        f"{model.lower()} decoder"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = (
        f"{metric_filename}_vs_embedding_dimension_"
        f"{model_filename}"
    )

    plt.savefig(
        f"{filename}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# -----------------------
# Generate all four plots
# -----------------------
plot_result("Frozen", "RMSE")
plot_result("Frozen", "R2")
plot_result("Fine-tuned", "RMSE")
plot_result("Fine-tuned", "R2")
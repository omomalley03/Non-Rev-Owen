from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "mcmaze"
BASE_CSV = OUT_DIR / "embedding_dim_velocity_mse.csv"
NO_CONV_CSV = OUT_DIR / "mlp_no_conv_embedding_dim_velocity_rmse.csv"
OUT_PNG = OUT_DIR / "embedding_dim_velocity_rmse_with_no_conv_mlp.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    records = read_csv(BASE_CSV)
    no_conv_records = [
        {
            "dimension": row["dimension"],
            "rmse_mean": row["rmse_mean"],
        }
        for row in read_csv(NO_CONV_CSV)
    ]

    colors = {
        "lambda=1": "#1f77b4",
        "lambda=0": "#d62728",
        "no_conv_mlp": "#2ca02c",
    }
    labels = {
        "lambda=1": "with reg",
        "lambda=0": "without reg",
        "no_conv_mlp": "MLP (no conv, yes reg)",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key in ["lambda=1", "lambda=0"]:
        series = sorted(
            (r for r in records if r["block_cca"] == key),
            key=lambda r: int(r["dimension"]),
        )
        ax.plot(
            [int(r["dimension"]) for r in series],
            [float(r["rmse_mean"]) for r in series],
            marker="o",
            linewidth=2,
            color=colors[key],
            label=labels[key],
        )

    no_conv_series = sorted(no_conv_records, key=lambda r: int(r["dimension"]))
    ax.plot(
        [int(r["dimension"]) for r in no_conv_series],
        [float(r["rmse_mean"]) for r in no_conv_series],
        marker="^",
        linewidth=2,
        color=colors["no_conv_mlp"],
        label=labels["no_conv_mlp"],
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Mean hand velocity RMSE (mm/s)")
    ax.set_title("MC Maze hand velocity prediction")
    ax.grid(True, which="major", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

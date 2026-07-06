from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "mcmaze"

RUNS = [
    {
        "block_cca": "lambda=1",
        "dimension": 16,
        "run": "mcmaze/runs/20260705_164826_d16_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=1",
        "dimension": 32,
        "run": "mcmaze/runs/20260705_180855_d32_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=1",
        "dimension": 64,
        "run": "mcmaze/runs/20260705_181142_d64_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=1",
        "dimension": 128,
        "run": "mcmaze/runs/20260705_183042_d128_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=1",
        "dimension": 256,
        "run": "mcmaze/runs/20260705_183549_d256_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=1",
        "dimension": 512,
        "run": "mcmaze/runs/20260705_184239_d512_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=1",
        "dimension": 1024,
        "run": "mcmaze/runs/20260705_184738_d1024_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=0",
        "dimension": 16,
        "run": "mcmaze/runs/20260705_185537_d16_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca0.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=0",
        "dimension": 32,
        "run": "mcmaze/runs/20260705_185627_d32_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca0.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=0",
        "dimension": 64,
        "run": "mcmaze/runs/20260705_185722_d64_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca0.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=0",
        "dimension": 128,
        "run": "mcmaze/runs/20260705_185825_d128_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca0.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=0",
        "dimension": 256,
        "run": "mcmaze/runs/20260705_185947_d256_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca0.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=0",
        "dimension": 512,
        "run": "mcmaze/runs/20260705_190145_d512_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca0.0_sig10.0_s1",
    },
    {
        "block_cca": "lambda=0",
        "dimension": 1024,
        "run": "mcmaze/runs/20260705_190457_d1024_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca0.0_sig10.0_s1",
    },
]


def read_mlp_metrics(run: str) -> dict[str, str]:
    path = ROOT / run / "outputs" / "velocity_prediction_100ms" / "metrics.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["model"] == "mlp":
            return row
    raise RuntimeError(f"No MLP metrics row found in {path}")


def main() -> None:
    records = []
    for item in RUNS:
        metrics = read_mlp_metrics(item["run"])
        mse_x = float(metrics["mse_x"])
        mse_y = float(metrics["mse_y"])
        records.append(
            {
                "block_cca": item["block_cca"],
                "dimension": item["dimension"],
                "mse_x": mse_x,
                "mse_y": mse_y,
                "mse_mean": (mse_x + mse_y) / 2.0,
                "rmse_mean": float(metrics["rmse_mean"]),
                "r2_mean": float(metrics["r2_mean"]),
                "corr_x": float(metrics["corr_x"]),
                "corr_y": float(metrics["corr_y"]),
                "run": str(ROOT / item["run"]),
            }
        )

    csv_path = OUT_DIR / "embedding_dim_velocity_mse.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    colors = {"lambda=1": "#1f77b4", "lambda=0": "#d62728"}
    labels = {
        "lambda=1": "with reg.",
        "lambda=0": "without reg.",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key in ["lambda=1", "lambda=0"]:
        series = sorted((r for r in records if r["block_cca"] == key), key=lambda r: r["dimension"])
        ax.plot(
            [r["dimension"] for r in series],
            [r["rmse_mean"] for r in series],
            marker="o",
            linewidth=2,
            color=colors[key],
            label=labels[key],
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
    fig.savefig(OUT_DIR / "embedding_dim_velocity_rmse.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {OUT_DIR / 'embedding_dim_velocity_rmse.png'}")


if __name__ == "__main__":
    main()

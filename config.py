import dataclasses
import os
from dataclasses import dataclass


def _fmt(x):
    """Format a float for use in a filename (e.g. 1e-3 → '1e-3', 0.1 → '0.1')."""
    s = f"{x:.0e}" if x != 0 and (abs(x) < 0.01 or abs(x) >= 1000) else str(x)
    return s.replace("+", "")


@dataclass
class Config:
    # --- data ---
    nwb_path: str = (
        "/Users/omomalley03/Documents/Dissertation/Data/000129/sub-Indy/"
        "sub-Indy_desc-train_behavior+ecephys.nwb"
    )
    bin_ms: int = 5                     # resampling bin width (ms)
    sigma_ms: float = 10.0             # Gaussian smoothing std (ms)
    # All 1080 trials are exactly 120 bins (600 ms); each trial = one window.
    window_size: int = 120             # timesteps per window (bins)
    window_strategy: str = "trial_aligned"
    val_split: float = 0.1
    seed: int = 0

    # --- model ---
    d: int = 10                       # embedding dimension (per snapshot)
    hidden_dim: int = 256              # MLP hidden layer width
    depth: int = 3                     # number of MLP layers

    # --- training ---
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_bt: float = 5e-9            # Barlow Twins covariance regularisation weight default 5e-3

    # --- LR scheduler (CosineAnnealingWarmRestarts) ---
    T_0: int = 10
    T_mult: int = 2

    # --- I/O ---
    ckpt_dir: str = "checkpoints"
    out_dir: str = "outputs"

    def run_name(self) -> str:
        """Short descriptive tag encoding the key hyperparameters."""
        return (
            f"d{self.d}_h{self.hidden_dim}_dep{self.depth}"
            f"_bs{self.batch_size}_ep{self.epochs}"
            f"_lr{_fmt(self.lr)}_lbt{_fmt(self.lambda_bt)}"
            f"_sig{self.sigma_ms}_s{self.seed}"
        )

    def save_about(self, run_dir: str) -> None:
        """Write a human-readable about.txt with all hyperparameters."""
        os.makedirs(run_dir, exist_ok=True)
        fields = dataclasses.fields(self)
        groups = {
            "data":     ["nwb_path", "bin_ms", "sigma_ms", "window_size",
                         "window_strategy", "val_split", "seed"],
            "model":    ["d", "hidden_dim", "depth"],
            "training": ["batch_size", "epochs", "lr", "weight_decay", "lambda_bt"],
            "scheduler":["T_0", "T_mult"],
        }
        name_to_val = {f.name: getattr(self, f.name) for f in fields}
        lines = [f"run_dir : {run_dir}\n"]
        for group, keys in groups.items():
            lines.append(f"[{group}]")
            for k in keys:
                lines.append(f"  {k:<20} = {name_to_val[k]}")
            lines.append("")
        with open(os.path.join(run_dir, "about.txt"), "w") as f:
            f.write("\n".join(lines))

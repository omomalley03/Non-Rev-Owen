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
        "/Users/omomalley03/Documents/Dissertation/Data/000128/sub-Jenkins/"
        "sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
    )
    bin_ms: int = 10                     # resampling bin width (ms)
    sigma_ms: float = 10.0              # Gaussian smoothing std (ms)
    softnorm_method: str = "churchland"       # 'churchland', 'max', or 'none' (SCA-style per-neuron softnorm)
    # MC_Maze (Jenkins): trials are variable length (~2-4 s).  Each window is
    # aligned to align_field, taking pre_ms before and post_ms after, so
    # window_size = (pre_ms + post_ms) / bin_ms.
    align_field: str = "move_onset_time"
    pre_ms: int = 200                   # ms before align_field
    post_ms: int = 140                  # ms after align_field
    window_size: int = 90              # = (pre_ms + post_ms) / bin_ms
    window_strategy: str = "trial_aligned"
    val_split: float = 0.1              # only used if dataset has no `split` column
    seed: int = 0
    split: str = "dataset"                   # "random" or "dataset" (use `split` column if present, else random split)
    synth_data_path: str = "/Users/omomalley03/Documents/Dissertation/POC_MLP/rotations_4planes.npy"


    # USE THIS FOR HPC
    # synth_data_path: str = os.environ.get( 
    #     "SYNTH_DATA_PATH",
    #     "/Volumes/ADATA HD710/data_owen/FACED/processed/faced_data.npy",
    # )
    synth_noise_std: float = 0

    # --- model ---
    d: int = 4                       # embedding dimension (per snapshot)
    hidden_dim: int = 256              # MLP hidden layer width
    depth: int = 3                     # number of MLP layers (1 = pure linear, SCA-equivalent)
    dropout: float = 0.2            # dropout probability applied after each hidden activation
    temporal_filters: int = 0         # symmetric temporal conv front-end; 0 disables it
    temporal_kernel_size: int = 31     # odd; ±15 ms at 1 ms sampling (tunable; sweep e.g. 15/31/51)

    F_mean_axis: tuple = (0,2) # (0,2) to zero-mean per dim across batch and time, (0,) to zero-mean per dim across batch only, None or () for no internal mean-centering before Barlow Twins term
    # --- training ---
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_xp: float = 0.0              # cross-plane non-reversibility regularizer weight
    lambda_bt: float = 0.0              # Barlow Twins covariance regularizer weight
    lambda_plane_bt: float = 0.0         # plane-aware BT: allow within-plane covariance, penalize cross-plane covariance
    lambda_block_cca: float = 1.0        # plane-level linear redundancy penalty
    lambda_start_frac: float = 1.0       # linear lambda warm-up: fraction of full lambda at epoch 1,
                                         # ramping linearly to 1.0 (full lambda) at the final epoch.
                                         # 1.0 = no warm-up (full lambda throughout)

    s_objective: str = "mean"      # "sum" keeps old -S; "softmin" focuses the weakest plane
    s_softmin_tau: float = 0.05          # lower values focus harder on the weakest plane
    # plane_dropout_p: float = 0.0       # unused experiment: randomly drop whole planes in the S objective
    block_cca_eps: float = 1e-6
    # logdet_eps: float = 1e-4           # unused experiment
    # predict_ridge: float = 1e-3        # unused experiment
    # predict_adv_hidden: int = 64       # unused experiment
    # predict_adv_lr: float = 1e-3       # unused experiment
    # predict_adv_steps: int = 1         # unused experiment
    # predict_adv_max_samples: int = 512 # unused experiment

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
            f"_lr{_fmt(self.lr)}_lxp{_fmt(self.lambda_xp)}_lbt{_fmt(self.lambda_bt)}"
            f"_lcca{_fmt(self.lambda_block_cca)}"
            f"_sig{self.sigma_ms}_s{self.seed}"
        )

    def save_about(self, run_dir: str) -> None:
        """Write a human-readable about.txt with all hyperparameters."""
        os.makedirs(run_dir, exist_ok=True)
        fields = dataclasses.fields(self)
        groups = {
            "data":     ["nwb_path", "bin_ms", "sigma_ms", "softnorm_method",
                         "align_field", "pre_ms", "post_ms", "window_size",
                         "window_strategy", "val_split", "seed",
                         "synth_data_path", "synth_noise_std"],
            "model":    ["d", "hidden_dim", "depth", "dropout",
                         "temporal_filters", "temporal_kernel_size"],
            "training": ["batch_size", "epochs", "lr", "weight_decay",
                         "lambda_xp", "lambda_bt", "lambda_plane_bt",
                         "lambda_block_cca", "lambda_start_frac",
                         "s_objective", "s_softmin_tau", "block_cca_eps"],
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

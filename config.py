import dataclasses
import os
from dataclasses import dataclass

from paths import DATASET_NAME, RUNS_BASE


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_float_tuple(name: str, default: tuple = ()) -> tuple:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _fmt(x):
    """Format a float for use in a filename (e.g. 1e-3 → '1e-3', 0.1 → '0.1')."""
    s = f"{x:.0e}" if x != 0 and (abs(x) < 0.01 or abs(x) >= 1000) else str(x)
    return s.replace("+", "")


@dataclass
class Config:
    # --- data ---
    dataset_name: str = DATASET_NAME
    nwb_path: str = os.environ.get(
        "NWB_PATH",
        os.path.join(RUNS_BASE, "sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"),
    )
    bin_ms: int = _env_int("BIN_MS", 10)                     # resampling bin width (ms)
    sigma_ms: float = _env_float("SIGMA_MS", 10.0)              # Gaussian smoothing std (ms)
    softnorm_method: str = _env_str("SOFTNORM_METHOD", "churchland")       # 'churchland', 'max', or 'none' (SCA-style per-neuron softnorm)
    # MC_Maze (Jenkins): trials are variable length (~2-4 s).  Each window is
    # aligned to align_field, taking pre_ms before and post_ms after, so
    # window_size = (pre_ms + post_ms) / bin_ms.
    align_field: str = _env_str("ALIGN_FIELD", "move_onset_time")
    pre_ms: int = _env_int("PRE_MS", 200)                   # ms before align_field
    post_ms: int = _env_int("POST_MS", 140)                  # ms after align_field
    window_size: int = _env_int("WINDOW_SIZE", 90)              # = (pre_ms + post_ms) / bin_ms
    window_strategy: str = _env_str("WINDOW_STRATEGY", "trial_aligned")
    val_split: float = _env_float("VAL_SPLIT", 0.1)  # only used if dataset has no `split` column
    seed: int = _env_int("SEED", 1)
    split: str = _env_str("SPLIT", "dataset")                   # "random" or "dataset" (use `split` column if present, else random split)
    synth_data_path: str = os.environ.get(
        "SYNTH_DATA_PATH",
        os.path.join(RUNS_BASE, "rotations_mixed_freqs.npy"),
    )
    synth_data_layout: str = os.environ.get("SYNTH_DATA_LAYOUT", "auto")  # auto, knt, ktn
    synth_normalize: str = os.environ.get("SYNTH_NORMALIZE", "none")      # none, zscore
    synth_preprocess: str = _env_str("SYNTH_PREPROCESS", "none")          # none, car, analytic_bandpass
    eeg_fs: float = _env_float("EEG_FS", 250.0)
    eeg_bands: str = _env_str("EEG_BANDS", "theta:4-8,alpha:8-13,beta:13-30,gamma:30-45")
    synth_noise_std: float = _env_float("SYNTH_NOISE_STD", 0.2)
    synth_max_trials: int = _env_int("SYNTH_MAX_TRIALS", 0)                # 0 = all trials
    synth_split: str = _env_str("SYNTH_SPLIT", "random")                   # random, train_eq_val, or subject_random
    synth_labels_path: str = _env_str("SYNTH_LABELS_PATH", os.environ.get("PHYSIONETMI_LABELS_NPY", ""))
    synth_subjects_path: str = _env_str("SYNTH_SUBJECTS_PATH", "")
    synth_subject_count: int = _env_int("SYNTH_SUBJECT_COUNT", 0)           # 0 = all subjects
    synth_subject_ids: str = _env_str("SYNTH_SUBJECT_IDS", "")             # comma-separated explicit subject ids
    synth_holdout_subject_count: int = _env_int("SYNTH_HOLDOUT_SUBJECT_COUNT", 0)  # excluded from train/val
    synth_holdout_subject_ids: str = _env_str("SYNTH_HOLDOUT_SUBJECT_IDS", "")     # explicit held-out subjects
    synth_viz_max_trials: int = _env_int("SYNTH_VIZ_MAX_TRIALS", 64)
    synth_viz_max_timepoints: int = _env_int("SYNTH_VIZ_MAX_TIMEPOINTS", 400)
    synth_viz_participant_mode: str = _env_str("SYNTH_VIZ_PARTICIPANT_MODE", "top_zeta")  # top_zeta or random
    synth_viz_participant_count: int = _env_int("SYNTH_VIZ_PARTICIPANT_COUNT", 4)

    # --- model ---
    d: int = _env_int("D", 2)                       # embedding dimension (per snapshot)
    hidden_dim: int = _env_int("HIDDEN_DIM", 64)              # MLP hidden layer width
    depth: int = _env_int("DEPTH", 1)                     # number of MLP layers (1 = pure linear, SCA-equivalent)
    dropout: float = _env_float("DROPOUT", 0.2)            # dropout probability applied after each hidden activation
    temporal_filters: int = _env_int("TEMPORAL_FILTERS", 0)        # per-channel temporal filters; 0 disables the front-end
    temporal_kernel_size: int = _env_int("TEMPORAL_KERNEL_SIZE", 61)     # odd; zero-phase 'same' conv (tunable; sweep e.g. 15/31/51)
    temporal_frontend: str = _env_str("TEMPORAL_FRONTEND", "symmetric")  # symmetric, multiscale_symmetric, mixed_parity, or residual
    residual_kernels: str = _env_str("RESIDUAL_KERNELS", "3,7,15,31")    # comma-separated odd kernels for multi-scale front-ends
    multiscale_symmetric_conv_layers: int = _env_int("MULTISCALE_SYMMETRIC_CONV_LAYERS", 1)  # 1 or 2; only for multiscale_symmetric
    antisymmetric_planes: int = _env_int("ANTISYMMETRIC_PLANES", -1)      # mixed_parity only; -1 auto-selects half the planes

    F_mean_axis: tuple = (0,2) # (0,2) to zero-mean per dim across batch and time, (0,) to zero-mean per dim across batch only, None or () for no internal mean-centering before Barlow Twins term
    # --- training ---
    batch_size: int = _env_int("BATCH_SIZE", 256)
    epochs: int = _env_int("EPOCHS", 80)
    lr: float = _env_float("LR", 5e-4)
    weight_decay: float = _env_float("WEIGHT_DECAY", 1e-4)
    lambda_xp: float = _env_float("LAMBDA_XP", 0.0)              # cross-plane non-reversibility regularizer weight
    lambda_bt: float = _env_float("LAMBDA_BT", 0.0)              # Barlow Twins covariance regularizer weight
    lambda_plane_bt: float = _env_float("LAMBDA_PLANE_BT", 0.0)         # plane-aware BT: allow within-plane covariance, penalize cross-plane covariance
    lambda_block_cca: float = _env_float("LAMBDA_BLOCK_CCA", 0.0)        # plane-level linear redundancy penalty
    lambda_start_frac: float = _env_float("LAMBDA_START_FRAC", 1.0)       # linear lambda warm-up: fraction of full lambda at epoch 1,
                                         # ramping linearly to 1.0 (full lambda) at the final epoch.
                                         # 1.0 = no warm-up (full lambda throughout)
    val_checkpoint_metric: str = _env_str("VAL_CHECKPOINT_METRIC", "zeta")  # "zeta" or "s"
    val_checkpoint_thresholds: tuple = _env_float_tuple("VAL_CHECKPOINTS", ())
    val_zeta_checkpoint_thresholds: tuple = _env_float_tuple(
        "VAL_ZETA_CHECKPOINTS",
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    val_s_checkpoint_thresholds: tuple = _env_float_tuple("VAL_S_CHECKPOINTS", ())
    checkpoint_every_epochs: int = _env_int("CHECKPOINT_EVERY_EPOCHS", 0)

    s_objective: str = _env_str("S_OBJECTIVE", "mean")      # "sum" keeps old -S; "softmin" focuses the weakest plane
    s_softmin_tau: float = _env_float("S_SOFTMIN_TAU", 0.05)          # lower values focus harder on the weakest plane
    # plane_dropout_p: float = 0.0       # unused experiment: randomly drop whole planes in the S objective
    block_cca_eps: float = 1e-6
    # logdet_eps: float = 1e-4           # unused experiment
    # predict_ridge: float = 1e-3        # unused experiment
    # predict_adv_hidden: int = 64       # unused experiment
    # predict_adv_lr: float = 1e-3       # unused experiment
    # predict_adv_steps: int = 1         # unused experiment
    # predict_adv_max_samples: int = 512 # unused experiment

    # --- LR scheduler (CosineAnnealingWarmRestarts) ---
    T_0: int = _env_int("T_0", 10)
    T_mult: int = _env_int("T_MULT", 2)

    # --- I/O ---
    ckpt_dir: str = "checkpoints"
    out_dir: str = "outputs"

    def __post_init__(self) -> None:
        frontend = (self.temporal_frontend or "").lower()
        if frontend in {
            "mixed_parity",
            "mixed_symmetric_antisymmetric",
            "mixed_sym_anti",
            "sym_anti",
        } and self.antisymmetric_planes < 0:
            if self.d % 2 != 0:
                raise ValueError(f"mixed_parity requires an even embedding dimension, got d={self.d}")
            self.antisymmetric_planes = max(1, (self.d // 2) // 2)

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
            "data":     ["dataset_name", "nwb_path", "bin_ms", "sigma_ms", "softnorm_method",
                         "align_field", "pre_ms", "post_ms", "window_size",
                         "window_strategy", "val_split", "seed",
                         "synth_data_path", "synth_data_layout", "synth_normalize",
                         "synth_preprocess", "eeg_fs", "eeg_bands",
                         "synth_noise_std", "synth_max_trials", "synth_split",
                         "synth_labels_path", "synth_subjects_path", "synth_subject_count", "synth_subject_ids",
                         "synth_holdout_subject_count", "synth_holdout_subject_ids",
                         "synth_viz_max_trials", "synth_viz_max_timepoints",
                         "synth_viz_participant_mode", "synth_viz_participant_count"],
            "model":    ["d", "hidden_dim", "depth", "dropout",
                         "temporal_filters", "temporal_kernel_size",
                         "temporal_frontend", "residual_kernels",
                         "multiscale_symmetric_conv_layers",
                         "antisymmetric_planes"],
            "training": ["batch_size", "epochs", "lr", "weight_decay",
                         "lambda_xp", "lambda_bt", "lambda_plane_bt",
                         "lambda_block_cca", "lambda_start_frac",
                         "s_objective", "s_softmin_tau", "block_cca_eps",
                         "val_checkpoint_metric", "val_checkpoint_thresholds",
                         "val_zeta_checkpoint_thresholds",
                         "val_s_checkpoint_thresholds",
                         "checkpoint_every_epochs"],
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

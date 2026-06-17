
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from config import Config
from paths import RUNS_DIR
from data import load_mcmaze_cached, gaussian_smooth, soft_normalize, make_windows, train_val_split
from model import MLP
from train import train
from visualize import make_diagnostic_plots, _hand_windows_from_raw
from best_metrics import append_best_model_metrics

import matplotlib
matplotlib.use("macosx")   # visualize.py forces "Agg" on import; switch back for interactive display
import matplotlib.pyplot as plt

cfg = Config()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RUNS_DIR, f"{timestamp}_{cfg.run_name()}")
cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
cfg.out_dir  = os.path.join(run_dir, "outputs")
cfg.save_about(run_dir)
print(f"Run directory: {run_dir}")


print("Loading MC_Maze data …")
spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = load_mcmaze_cached(
    cfg.nwb_path, cfg.bin_ms
)
N = spikes_raw.shape[0]
print(f"  Channels: {N}  |  Bin width: {cfg.bin_ms} ms  |  Trials: {len(trial_info)}")

sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
print(f"  Gaussian smoothing sigma: {cfg.sigma_ms} ms = {sigma_samples} bins")
X_smooth = gaussian_smooth(spikes_raw, sigma_samples)

if cfg.softnorm_method and cfg.softnorm_method != "none":
    print(f"  Soft-normalising per neuron (method={cfg.softnorm_method!r})")
    X_smooth = soft_normalize(X_smooth, method=cfg.softnorm_method)

print(f"Windowing ({cfg.window_strategy}, align={cfg.align_field}, "
        f"pre={cfg.pre_ms}ms, T={cfg.window_size}) …")
windows = make_windows(
    X_smooth, trial_info, time_index_s, bin_width_s,
    strategy=cfg.window_strategy, window_size=cfg.window_size,
    align_field=cfg.align_field, pre_ms=cfg.pre_ms,
)

# plt.imshow(windows[0,80:110,:], cmap="Greys", vmin=0)
# plt.show()

# --- GELU activation (black line, white background, no grid) ---
x = torch.linspace(-4, 4, 400)
y = torch.nn.functional.gelu(x)

fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy(), color="black", linewidth=6)
ax.grid(False)
plt.show()
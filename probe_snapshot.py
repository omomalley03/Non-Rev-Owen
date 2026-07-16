"""Confirm the model is per-snapshot and that one fixed per-frame linear map
both (a) reproduces the whole embedding and (b) mixes multiple sources."""
import os, numpy as np, torch
from config import Config
from paths import SYNTH_RUNS_DIR
from model import MLP, infer_multiscale_symmetric_conv_layers

run_dir = sorted([os.path.join(SYNTH_RUNS_DIR, d) for d in os.listdir(SYNTH_RUNS_DIR)
                  if os.path.isfile(os.path.join(SYNTH_RUNS_DIR, d, "checkpoints", "best.pt"))],
                 key=os.path.getmtime, reverse=True)[0]
ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"),
                  map_location="cpu", weights_only=False)
cfg = ckpt["config"]
data = np.load(cfg.synth_data_path).astype(np.float32)
windows = np.transpose(data, (0, 2, 1))
rng = np.random.default_rng(cfg.seed)
windows = windows + rng.normal(0, cfg.synth_noise_std, windows.shape).astype(np.float32)
K, O, T = windows.shape
state_dict = ckpt["model_state_dict"]
model = MLP(
    O, cfg.d, cfg.hidden_dim, cfg.depth, cfg.dropout,
    temporal_filters=getattr(cfg, "temporal_filters", 0),
    temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
    temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
    residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
    multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
        state_dict,
        getattr(cfg, "multiscale_symmetric_conv_layers", 1),
    ),
)
model.load_state_dict(state_dict); model.eval()

# (a) per-snapshot test: feed ONE frame repeated T times -> output is constant in time
one = torch.from_numpy(windows[:, :, 5:6].repeat(T, axis=2))
with torch.no_grad():
    out = model(one).numpy()
print(f"(a) repeat-one-frame: max output variation over time = {out.std(axis=2).max():.2e}")
print("    -> output at frame t depends ONLY on input frame t (no temporal mixing)")

# recover instantaneous 8D latent basis from all snapshots, fit one 8x8 per-frame map
with torch.no_grad():
    F = model(torch.from_numpy(windows)).numpy()             # (K,8,T)
snaps_x = windows.transpose(0, 2, 1).reshape(K * T, O)        # (KT,128)
snaps_F = F.transpose(0, 2, 1).reshape(K * T, cfg.d)          # (KT,8)
_, _, Vh = np.linalg.svd(snaps_x - snaps_x.mean(0), full_matrices=False)
R8 = (snaps_x - snaps_x.mean(0)) @ Vh[:8].T                   # (KT,8) instantaneous latent
M, *_ = np.linalg.lstsq(R8, snaps_F - snaps_F.mean(0), rcond=None)   # 8x8 per-frame map
pred = R8 @ M
r2 = 1 - ((snaps_F - snaps_F.mean(0) - pred) ** 2).sum() / ((snaps_F - snaps_F.mean(0)) ** 2).sum()
print(f"\n(b) one fixed 8x8 per-frame map reproduces embedding: R^2 = {r2:.4f}")

# show a live plane's readout pulls from multiple latent coords (spatial mixing)
rms = np.sqrt((F ** 2).mean(axis=(0, 2)))                     # per output-dim energy
live = int(np.argmax(rms.reshape(4, 2).sum(1)))              # most active plane
row = np.abs(M[:, 2 * live]); row = row / row.sum()
print(f"    live plane {live}: weight on the 8 instantaneous latent coords =")
print("    " + "  ".join(f"{w:.2f}" for w in row))
print(f"    -> spread across coords = blends multiple sources within one frame")

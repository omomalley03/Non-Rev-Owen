"""
Same probe, but on the ACTUAL trained checkpoint + real data, with the
4 latent sources recovered from the data via SVD (seed-free).

Mechanism under test:
  embedding plane p, trial k, as a complex signal, is
      z_kp(t) = G_kp e^{+it} + H_kp e^{-it}
  |G| = circle radius, |H| = ellipse defect.
  Hypothesis: G_kp = sum_j alpha_pj S_kj, a linear mix of the 4 same-
  frequency sources S_kj = (amplitude) e^{i phi_kj}, so the per-trial
  radius |G_kp| varies by constructive/destructive interference.
"""
import os
import numpy as np
import torch

from config import Config
from paths import SYNTH_RUNS_DIR
from model import MLP

runs = sorted(
    [os.path.join(SYNTH_RUNS_DIR, d) for d in os.listdir(SYNTH_RUNS_DIR)
     if os.path.isfile(os.path.join(SYNTH_RUNS_DIR, d, "checkpoints", "best.pt"))],
    key=os.path.getmtime, reverse=True,
)
run_dir = runs[0]
print(f"run: {os.path.basename(run_dir)}")
ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"),
                  map_location="cpu", weights_only=False)
cfg: Config = ckpt["config"]

# load data exactly like the pipeline
data = np.load(cfg.synth_data_path).astype(np.float32)
windows = np.transpose(data, (0, 2, 1))                 # (K, O, T)
if getattr(cfg, "synth_noise_std", 0.0) > 0:
    rng = np.random.default_rng(cfg.seed)
    windows = windows + rng.normal(0, cfg.synth_noise_std, windows.shape).astype(np.float32)
K, O, T = windows.shape
print(f"data {cfg.synth_data_path}  windows {windows.shape}")

model = MLP(in_channels=O, d=cfg.d, hidden_dim=cfg.hidden_dim,
            depth=cfg.depth, dropout=cfg.dropout)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

X = torch.from_numpy(windows)
with torch.no_grad():
    F = model(X)
    F_hat = (F - F.mean(dim=cfg.F_mean_axis, keepdim=True)).numpy()  # (K,d,T)

ts = np.linspace(0, 2 * np.pi, T)
e_plus = np.exp(1j * ts)
D = cfg.d // 2

# ── phasor decomposition of each embedding plane ──────────────────────────────
G = np.zeros((K, D), dtype=complex)
H = np.zeros((K, D), dtype=complex)
for p in range(D):
    z = F_hat[:, 2*p, :] + 1j * F_hat[:, 2*p+1, :]
    G[:, p] = (z * np.conj(e_plus)).mean(axis=1)
    H[:, p] = (z * e_plus).mean(axis=1)
radius = np.abs(G)

# ── recover the 4 sources from the DATA (seed-free) ───────────────────────────
# input-channel phasor at the fundamental: S_in[k,o] = sum_j alpha_jo e^{i phi_kj}
S_in = (windows * np.conj(e_plus)[None, None, :]).mean(axis=2)   # (K, O) complex
U, s, Vh = np.linalg.svd(S_in - S_in.mean(0, keepdims=True), full_matrices=False)
print(f"\ninput phasor singular values (top 8): "
      f"{np.round(s[:8] / s[0], 3)}")
sources = U[:, :4] * s[:4]            # (K,4) per-trial source amplitudes (basis of truth)

print("\n── per-plane: radius spread, circle-ness, interference fit ──")
print(f"{'plane':>5} {'radius rng':>14} {'CV':>6} {'|H|/|G|':>8} {'R^2(4 src)':>11}")
for p in range(D):
    cv = radius[:, p].std() / (radius[:, p].mean() + 1e-12)
    ell = np.abs(H[:, p]).mean() / (np.abs(G[:, p]).mean() + 1e-12)
    alpha, *_ = np.linalg.lstsq(sources, G[:, p], rcond=None)
    pred = sources @ alpha
    r2 = 1 - np.abs(G[:, p] - pred).var() / (np.abs(G[:, p] - G[:, p].mean()).var() + 1e-12)
    print(f"{p:>5} {radius[:,p].min():5.2f}-{radius[:,p].max():5.2f}  "
          f"{cv:6.3f} {ell:8.3f} {r2:11.3f}")

# ── counterfactual radius spread if each plane were ONE pure source ───────────
single = np.abs(sources[:, 0])
print(f"\nclean-separation radius CV (one source/plane): "
      f"{single.std()/single.mean():.4f}")
print(f"actual mean radius CV across planes:            "
      f"{(radius.std(0)/radius.mean(0)).mean():.4f}")

# ── does interference-predicted radius match the measured radius? ─────────────
# fit all planes jointly, compare predicted |G| to measured |G|
pred_all = np.zeros_like(G)
for p in range(D):
    a, *_ = np.linalg.lstsq(sources, G[:, p], rcond=None)
    pred_all[:, p] = sources @ a
corr = np.corrcoef(np.abs(pred_all).ravel(), radius.ravel())[0, 1]
print(f"\ncorr( interference-predicted radius , measured radius ) = {corr:.4f}")

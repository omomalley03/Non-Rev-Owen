"""
Probe: are the concentric circles in the embedding explained by
phasor interference of mixed same-frequency latents?

Plan:
  1. Generate synthetic data the same way as gen_synth, but KEEP the phases.
  2. Train the real depth-1 model with the real loss (pure S-maximization).
  3. Decompose each output plane's per-trial trajectory into phasors:
        z_kp(t) = G_kp e^{+it} + H_kp e^{-it}
     |G| = circle part, |H| = ellipse part.
  4. Test the interference prediction G_kp = sum_j alpha_pj e^{i phi_kj}
     by least-squares regressing G on the ground-truth phase phasors.
  5. Compare measured radius spread to the clean-separation counterfactual.
"""
import numpy as np
import torch

from config import Config
from model import MLP
from loss import non_reversibility_S, _batch_rms_normalize

K, T, O = 200, 100, 8 * 16  # 128
SEED = 0
NOISE = 0.2

rng = np.random.default_rng(SEED)
ts = np.linspace(0, 2 * np.pi, T)


def gen_with_phases():
    """Return data (K,T,O), latent r (K,T,8), phases (K,4), mixing C (8,O)."""
    phases = rng.uniform(0, 2 * np.pi, (K, 4))          # (K,4) one per latent
    lat = []
    for j in range(4):
        ang = ts[None, :] + phases[:, j:j+1]            # (K,T)
        lat += [np.cos(ang), np.sin(ang)]
    r = np.stack(lat, axis=2)                            # (K,T,8)
    C = rng.standard_normal((8, O)) / np.sqrt(8)
    x = r @ C                                            # (K,T,O)
    return x.astype(np.float32), r, phases, C


x, r, phases, C = gen_with_phases()
noise = rng.normal(0, NOISE, x.shape).astype(np.float32)
x_noisy = x + noise

# windows as (K, N, T) like the pipeline
windows = np.transpose(x_noisy, (0, 2, 1))              # (K, O, T)
X = torch.from_numpy(windows)

# ── train the real depth-1 model with the real loss ───────────────────────────
cfg = Config()
cfg.d = 8
cfg.depth = 1
cfg.lambda_xp = 0.0
cfg.lambda_bt = 0.0
torch.manual_seed(0)
model = MLP(in_channels=O, d=8, hidden_dim=256, depth=1, dropout=0.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(600):
    opt.zero_grad()
    F = model(X)
    F = F - F.mean(dim=(0, 2), keepdim=True)
    loss = -non_reversibility_S(_batch_rms_normalize(F))
    loss.backward()
    opt.step()
print(f"final S_ratio = {(-loss).item():.4f}")

# ── embeddings exactly as visualize_synth plots them ──────────────────────────
with torch.no_grad():
    F = model(X)
    F_hat = (F - F.mean(dim=(0, 2), keepdim=True)).numpy()   # (K,8,T)

# ── phasor decomposition per (trial, plane) ───────────────────────────────────
e_plus = np.exp(1j * ts)                                # e^{+it}
D = 4
G = np.zeros((K, D), dtype=complex)   # circle (rotation) part
H = np.zeros((K, D), dtype=complex)   # anti-rotation (ellipse) part
for p in range(D):
    z = F_hat[:, 2*p, :] + 1j * F_hat[:, 2*p+1, :]     # (K,T) complex plane signal
    G[:, p] = (z * np.conj(e_plus)).mean(axis=1)        # project onto e^{+it}
    H[:, p] = (z * e_plus).mean(axis=1)                 # project onto e^{-it}

radius = np.abs(G)                                      # (K,D) circle radius per trial
ellipse_frac = np.abs(H) / (np.abs(G) + 1e-12)          # 0 = perfect circle

print("\n── circle vs ellipse ──")
print(f"mean |H|/|G| (0=perfect circle): {ellipse_frac.mean():.3f}")

print("\n── radius spread across trials, per plane ──")
for p in range(D):
    cv = radius[:, p].std() / radius[:, p].mean()
    print(f"plane {p}: radius {radius[:,p].min():.2f}–{radius[:,p].max():.2f}  CV={cv:.3f}")

# ── test interference model: G[:,p] = sum_j alpha_pj e^{i phi_j} ──────────────
Phi = np.exp(1j * phases)                               # (K,4) latent phasors
print("\n── interference model fit  G_p ~ sum_j alpha_pj e^{i phi_kj} ──")
for p in range(D):
    alpha, *_ = np.linalg.lstsq(Phi, G[:, p], rcond=None)   # (4,) complex coeffs
    pred = Phi @ alpha
    ss_res = np.abs(G[:, p] - pred).var()
    ss_tot = np.abs(G[:, p] - G[:, p].mean()).var()
    r2 = 1 - ss_res / ss_tot
    contrib = np.abs(alpha)
    contrib = contrib / contrib.sum()
    mix = "  ".join(f"L{j}:{c:.2f}" for j, c in enumerate(contrib))
    print(f"plane {p}: R^2={r2:.3f}   latent mix [{mix}]")

# ── counterfactual: clean separation (one latent per plane) ───────────────────
# radius would be a fixed |alpha| + only noise jitter -> tiny CV
print("\n── counterfactual: if each plane were ONE pure latent ──")
clean_cv = []
for p in range(D):
    z = np.exp(1j * (ts[None, :] + phases[:, p:p+1]))   # single latent, unit amp
    g = (z * np.conj(e_plus)).mean(axis=1)
    clean_cv.append(np.abs(g).std() / np.abs(g).mean())
print(f"mean radius CV under clean separation: {np.mean(clean_cv):.4f}")
print(f"mean radius CV in the actual embedding: {(radius.std(0)/radius.mean(0)).mean():.4f}")

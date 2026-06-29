"""Fast iteration harness for the temporal front-end.

Goal: maximize the non-reversibility score S [mean/plane] on the val split of
rotations_mixed_freqs.npy by changing only the conv1d / temporal-filter design.

Run:  .venv/bin/python exp_temporal.py
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from loss import non_reversibility_S, _batch_rms_normalize, loss_fn

torch.set_num_threads(max(1, torch.get_num_threads()))


# ----------------------------- temporal front-ends -----------------------------

class SymMix(nn.Module):
    """groups=1 zero-phase conv: fully channel-mixing, out=filters (current design)."""
    def __init__(self, in_ch, filters, k):
        super().__init__()
        assert k % 2 == 1
        self.weight = nn.Parameter(torch.empty(filters, in_ch, k))
        self.bias = nn.Parameter(torch.zeros(filters))
        nn.init.uniform_(self.weight, -1.0 / (in_ch * k) ** 0.5, 1.0 / (in_ch * k) ** 0.5)
        self.pad = k // 2
        self.out_ch = filters

    def forward(self, x):
        w = self.weight + self.weight.flip(-1)
        return F.conv1d(x, w, self.bias, padding=self.pad)


class SymDepthwise(nn.Module):
    """groups=in_ch zero-phase conv: each input channel gets its own k filters.

    out_channels = in_ch * k.  Optionally per-output-channel scale normalization
    so every temporal feature is on the same scale before the MLP.
    """
    def __init__(self, in_ch, k_per_ch, k, norm=True):
        super().__init__()
        assert k % 2 == 1
        out_ch = in_ch * k_per_ch
        self.weight = nn.Parameter(torch.empty(out_ch, 1, k))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.uniform_(self.weight, -1.0 / k ** 0.5, 1.0 / k ** 0.5)
        self.pad = k // 2
        self.groups = in_ch
        self.out_ch = out_ch
        self.norm = nn.BatchNorm1d(out_ch, affine=True) if norm else None

    def forward(self, x):
        w = self.weight + self.weight.flip(-1)
        y = F.conv1d(x, w, self.bias, padding=self.pad, groups=self.groups)
        if self.norm is not None:
            y = self.norm(y)
        return y


class SymDepthwiseStack(nn.Module):
    """Two depthwise zero-phase convs with a nonlinearity between (per-channel).

    First conv expands each channel to k_per_ch features; second conv mixes those
    k_per_ch features within each original channel group (groups=in_ch) so no
    cross-channel mixing happens before the MLP.
    """
    def __init__(self, in_ch, k_per_ch, k, norm=True):
        super().__init__()
        assert k % 2 == 1
        mid = in_ch * k_per_ch
        self.w1 = nn.Parameter(torch.empty(mid, 1, k))
        self.b1 = nn.Parameter(torch.zeros(mid))
        self.w2 = nn.Parameter(torch.empty(mid, k_per_ch, k))   # groups=in_ch
        self.b2 = nn.Parameter(torch.zeros(mid))
        nn.init.uniform_(self.w1, -1.0 / k ** 0.5, 1.0 / k ** 0.5)
        nn.init.uniform_(self.w2, -1.0 / (k_per_ch * k) ** 0.5, 1.0 / (k_per_ch * k) ** 0.5)
        self.pad = k // 2
        self.in_ch = in_ch
        self.out_ch = mid
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(mid, affine=True) if norm else None

    def forward(self, x):
        w1 = self.w1 + self.w1.flip(-1)
        y = F.conv1d(x, w1, self.b1, padding=self.pad, groups=self.in_ch)
        y = self.act(y)
        w2 = self.w2 + self.w2.flip(-1)
        y = F.conv1d(y, w2, self.b2, padding=self.pad, groups=self.in_ch)
        if self.norm is not None:
            y = self.norm(y)
        return y


class SymMultiScale(nn.Module):
    """Per-channel zero-phase filters at several kernel sizes, concatenated.

    Captures both the slow (low-freq) and fast (high-freq) components present in
    the mixed-frequency data. out_ch = in_ch * k_per_ch * len(kernels).
    """
    def __init__(self, in_ch, k_per_ch, kernels=(7, 21, 41), norm=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.in_ch = in_ch
        for k in kernels:
            assert k % 2 == 1
            self.convs.append(SymDepthwise(in_ch, k_per_ch, k, norm=False))
        self.out_ch = in_ch * k_per_ch * len(kernels)
        self.norm = nn.BatchNorm1d(self.out_ch, affine=True) if norm else None

    def forward(self, x):
        y = torch.cat([c(x) for c in self.convs], dim=1)
        if self.norm is not None:
            y = self.norm(y)
        return y


FRONTENDS = {
    "mix": lambda N, args: SymMix(N, args.filters, args.k),
    "depthwise": lambda N, args: SymDepthwise(N, args.kpc, args.k, norm=False),
    "depthwise_norm": lambda N, args: SymDepthwise(N, args.kpc, args.k, norm=True),
    "depthwise_stack": lambda N, args: SymDepthwiseStack(N, args.kpc, args.k, norm=True),
    "multiscale": lambda N, args: SymMultiScale(N, args.kpc, norm=True),
}


class Model(nn.Module):
    def __init__(self, frontend, d=8, hidden=128, depth=2, dropout=0.0):
        super().__init__()
        self.frontend = frontend
        in_dim = frontend.out_ch
        layers = []
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, d))
        self.net = nn.Sequential(*layers)
        self.d = d
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):                       # (B, N, T)
        B = x.shape[0]
        x = self.frontend(x)                    # (B, C, T)
        C, T = x.shape[1], x.shape[2]
        x = x.permute(0, 2, 1).reshape(B * T, C)
        x = self.net(x)
        return x.reshape(B, T, self.d).permute(0, 2, 1)   # (B, d, T)


# ----------------------------------- train -------------------------------------

def run(name, frontend_fn, args, cfg, data):
    cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    N = data.shape[1]
    K = data.shape[0]
    n_val = int(K * cfg.val_split)
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(K, generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X = torch.from_numpy(data)
    Xtr, Xval = X[tr_idx], X[val_idx]

    model = Model(frontend_fn(N, args), d=cfg.d, hidden=cfg.hidden_dim,
                  depth=cfg.depth, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T_0, T_mult=cfg.T_mult)

    bs = args.batch
    best_val_s = -1e9
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0] - bs + 1, bs):
            b = Xtr[idx[i:i + bs]]
            opt.zero_grad()
            Fout = model(b)
            Fout = Fout - Fout.mean(dim=cfg.F_mean_axis, keepdim=True)
            loss = loss_fn(Fout, cfg=cfg, training=True)
            loss.backward()
            opt.step()
        sched.step(epoch)

        model.eval()
        with torch.no_grad():
            Fv = model(Xval)
            Fv = Fv - Fv.mean(dim=cfg.F_mean_axis, keepdim=True)
            s = non_reversibility_S(_batch_rms_normalize(Fv), "mean").item()
        if s > best_val_s:
            best_val_s = s
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"  [{name}] ep{epoch:3d}  valS={s:.4f}  best={best_val_s:.4f}")
    dt = time.time() - t0
    print(f"== {name:16s} params={n_params:>8,}  bestValS={best_val_s:.4f}  ({dt:.1f}s)")
    return best_val_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frontends", nargs="+", default=["mix", "depthwise", "depthwise_norm"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--filters", type=int, default=64)   # for mix
    p.add_argument("--kpc", type=int, default=4)         # filters per channel for depthwise
    p.add_argument("--k", type=int, default=31)          # kernel size
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = Config()
    data = np.transpose(np.load(cfg.synth_data_path).astype(np.float32), (0, 2, 1))
    print(f"data (K,N,T)={data.shape}  d={cfg.d}  depth={cfg.depth}  hidden={cfg.hidden_dim}")
    print(f"loss lambdas: bt={cfg.lambda_bt} cca={cfg.lambda_block_cca} xp={cfg.lambda_xp}  k={args.k} kpc={args.kpc} filters={args.filters}")
    results = {}
    for name in args.frontends:
        results[name] = run(name, FRONTENDS[name], args, cfg, data)
    print("\n==== summary (best val S, higher=better) ====")
    for name, s in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:16s} {s:.4f}")


if __name__ == "__main__":
    main()

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricConv1d(nn.Module):
    """Per-channel zero-phase temporal filter bank (depthwise).

    Each of the ``in_channels`` input channels gets its own ``filters_per_channel``
    temporal filters (grouped conv, ``groups=in_channels``) — no cross-channel
    mixing happens in the front-end, so the temporal features are extracted
    independently per channel, as intended.  Output has
    ``in_channels * filters_per_channel`` channels.

    The effective time kernel is palindromic (``w + w.flip(time)``), i.e. an exact
    zero-phase 'same' conv: the filter introduces no directional phase, so any
    non-reversibility in the output reflects genuine structure in the data rather
    than a phase-lead/lag artifact of the filter.

    A BatchNorm on the output puts every temporal feature on the same scale before
    the MLP, so no single channel/filter dominates.
    """

    def __init__(self, in_channels: int, filters_per_channel: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "use an odd kernel for an exact zero-phase 'same' conv"
        out_channels = in_channels * filters_per_channel
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.uniform_(self.weight, -1.0 / kernel_size ** 0.5, 1.0 / kernel_size ** 0.5)
        self.padding = kernel_size // 2
        self.groups = in_channels
        self.out_channels = out_channels
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, in, T)
        w = self.weight + self.weight.flip(-1)            # palindromic in time
        y = F.conv1d(x, w, self.bias, padding=self.padding, groups=self.groups)
        return self.norm(y)                               # (B, in*filters_per_channel, T)

class MLP(nn.Module):
    """Per-timepoint MLP embedder with shared weights across time.

    Input  x : (B, N, T)  — batch size, N channels, T timesteps
    Output   : (B, d, T)  — embedded trajectories

    """

    def __init__(self, in_channels: int, d: int = 128, hidden_dim: int = 256, depth: int = 3, dropout: float = 0.0,
                 temporal_filters: int = 0, temporal_kernel_size: int = 31):
        super().__init__()
        assert depth >= 1, "depth must be at least 1"

        if temporal_filters > 0:
            # per-channel filter bank: each input channel -> temporal_filters filters
            self.temporal_conv = SymmetricConv1d(in_channels, temporal_filters, temporal_kernel_size)
            in_channels = self.temporal_conv.out_channels   # use temporal features only (no raw concat)
        else:
            self.temporal_conv = None

        layers = []
        in_dim = in_channels
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d))   # final projection, no activation

        self.net = nn.Sequential(*layers)
        self.d = d
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        if self.temporal_conv is not None:
            # x = torch.cat([x, self.temporal_conv(x)], dim=1)   # (B, N + temporal_filters, T)
            x = self.temporal_conv(x)
        C = x.shape[1]
        x = x.permute(0, 2, 1).reshape(B * T, C)   # (B*T, C) -- each row is a single timepoint across all channels
        x = self.net(x)                              # (B*T, d)
        x = x.reshape(B, T, self.d).permute(0, 2, 1)  # (B, d, T)
        return x

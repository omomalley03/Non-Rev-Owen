import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricConv1d(nn.Module):
    """Conv1d whose effective time kernel is palindromic (zero-phase).

    Stores a free weight w and convolves with ``w + w.flip(time)``, which is
    symmetric by construction (``flip(w + flip(w)) == w + flip(w)``).  A
    symmetric kernel cannot phase-shift, so the filter carries no preferred
    direction of time and the temporal conv cannot manufacture non-reversibility
    on its own.  Channels are fully mixed (groups=1); only the time axis is
    constrained.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "use an odd kernel for an exact zero-phase 'same' conv"
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, in, T)
        w = self.weight + self.weight.flip(-1)            # palindromic in time
        return F.conv1d(x, w, self.bias, padding=self.padding)   # (B, out, T)


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
            self.temporal_conv = SymmetricConv1d(in_channels, temporal_filters, temporal_kernel_size)
            in_channels = in_channels + temporal_filters   # concat raw input with temporal features
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
            x = torch.cat([x, self.temporal_conv(x)], dim=1)   # (B, N + temporal_filters, T)
        C = x.shape[1]
        x = x.permute(0, 2, 1).reshape(B * T, C)   # (B*T, C) -- each row is a single timepoint across all channels
        x = self.net(x)                              # (B*T, d)
        x = x.reshape(B, T, self.d).permute(0, 2, 1)  # (B, d, T)
        return x

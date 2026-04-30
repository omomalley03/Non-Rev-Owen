import torch
import torch.nn as nn


class MLP(nn.Module):
    """Per-timepoint MLP embedder with shared weights across time.

    Input  x : (B, N, T)  — batch size, N channels, T timesteps
    Output   : (B, d, T)  — embedded trajectories

    """

    def __init__(self, in_channels: int, d: int = 128, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        assert depth >= 1, "depth must be at least 1"

        layers = []
        in_dim = in_channels
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
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
        x = x.permute(0, 2, 1).reshape(B * T, N)   # (B*T, N) -- each row is a single timepoint across all channels
        x = self.net(x)                              # (B*T, d)
        x = x.reshape(B, T, self.d).permute(0, 2, 1)  # (B, d, T)
        return x

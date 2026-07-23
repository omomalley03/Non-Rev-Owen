import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_kernels(kernels) -> tuple[int, ...]:
    if isinstance(kernels, str):
        parts = [p.strip() for p in kernels.split(",") if p.strip()]
        if not parts:
            raise ValueError("residual_kernels must contain at least one integer")
        return tuple(int(p) for p in parts)
    return tuple(int(k) for k in kernels)


def _split_dims(total_dim: int, num_splits: int) -> list[int]:
    if total_dim <= 0:
        raise ValueError("total_dim must be positive")
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")
    if total_dim < num_splits:
        raise ValueError("total_dim must be at least the number of splits")
    base_dim = total_dim // num_splits
    dims = [base_dim] * num_splits
    dims[-1] += total_dim - sum(dims)
    return dims


def infer_multiscale_symmetric_conv_layers(state_dict, default: int = 1) -> int:
    if any(
        key.startswith("temporal_conv.")
        and ".temporal_branches." in key
        and ".conv2." in key
        for key in state_dict
    ):
        return 2
    return int(default)


def _make_pointwise_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
) -> nn.Sequential:
    layers = []
    current_dim = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


def _make_odd_pointwise_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    depth: int,
) -> nn.Sequential:
    """Pointwise network satisfying g(-x) = -g(x)."""
    layers: list[nn.Module] = []
    current_dim = in_dim

    for _ in range(depth - 1):
        layers.extend([
            nn.Linear(current_dim, hidden_dim, bias=False),

            # LayerNorm without beta or gamma.
            # This satisfies LN(-x) = -LN(x).
            nn.LayerNorm(hidden_dim, elementwise_affine=False),

            # Odd activation
            nn.Tanh(),
        ])
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, out_dim, bias=False))
    return nn.Sequential(*layers)

def _apply_pointwise_net(x: torch.Tensor, net: nn.Module, out_dim: int) -> torch.Tensor:
    B, C, T = x.shape
    y = x.permute(0, 2, 1).reshape(B * T, C)
    y = net(y)
    return y.reshape(B, T, out_dim).permute(0, 2, 1)


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
        return y
        # return self.norm(y)                               # (B, in*filters_per_channel, T)


class SymmetricBranchConv1d(nn.Module):
    """One zero-phase depthwise branch for a single temporal scale."""

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernel_size: int,
        conv_layers: int = 1,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("symmetric branch kernels must be positive odd integers")
        conv_layers = int(conv_layers)
        if conv_layers not in {1, 2}:
            raise ValueError("symmetric branch conv_layers must be 1 or 2")
        out_channels = in_channels * filters_per_channel
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        nn.init.uniform_(self.conv.weight, -1.0 / kernel_size ** 0.5, 1.0 / kernel_size ** 0.5)
        nn.init.zeros_(self.conv.bias)
        self.conv2 = None
        self.activation = None
        if conv_layers == 2:
            self.activation = nn.GELU()
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
            )
            scale = (kernel_size * filters_per_channel) ** 0.5
            nn.init.uniform_(self.conv2.weight, -1.0 / scale, 1.0 / scale)
            nn.init.zeros_(self.conv2.bias)
        self.kernel = int(kernel_size)
        self.groups = int(in_channels)
        self.conv_layers = int(conv_layers)

    @staticmethod
    def _effective_weight(conv: nn.Conv1d) -> torch.Tensor:
        return conv.weight + conv.weight.flip(-1)

    def effective_weight(self, layer: int = 1) -> torch.Tensor:
        if layer == 1:
            return self._effective_weight(self.conv)
        if layer == 2 and self.conv2 is not None:
            return self._effective_weight(self.conv2)
        raise ValueError(f"expected layer 1 or 2, got {layer}")

    @property
    def weight(self):
        weights = [self.conv.weight.flatten()]
        if self.conv2 is not None:
            weights.append(self.conv2.weight.flatten())
        return torch.cat(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(
            x,
            self.effective_weight(1),
            self.conv.bias,
            padding=self.kernel // 2,
            groups=self.groups,
        )
        if self.conv2 is None:
            return y
        y = self.activation(y)
        return F.conv1d(
            y,
            self.effective_weight(2),
            self.conv2.bias,
            padding=self.kernel // 2,
            groups=self.groups,
        )

class AntiSymmetricBranchConv1d(nn.Module):
    """One derivative-like depthwise branch for a single temporal scale."""

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernel_size: int,
        conv_layers: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("anti-symmetric branch kernels must be positive odd integers")
        conv_layers = int(conv_layers)
        if conv_layers not in {1, 2}:
            raise ValueError("anti-symmetric branch conv_layers must be 1 or 2")
        out_channels = in_channels * filters_per_channel
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=bias,
        )
        nn.init.uniform_(self.conv.weight, -1.0 / kernel_size ** 0.5, 1.0 / kernel_size ** 0.5)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.conv2 = None
        self.activation = None
        if conv_layers == 2:
            self.activation = nn.GELU()
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=bias,
            )
            scale = (kernel_size * filters_per_channel) ** 0.5
            nn.init.uniform_(self.conv2.weight, -1.0 / scale, 1.0 / scale)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
        self.kernel = int(kernel_size)
        self.groups = int(in_channels)
        self.conv_layers = int(conv_layers)

    @staticmethod
    def _effective_weight(conv: nn.Conv1d) -> torch.Tensor:
        return conv.weight - conv.weight.flip(-1)

    def effective_weight(self, layer: int = 1) -> torch.Tensor:
        if layer == 1:
            return self._effective_weight(self.conv)
        if layer == 2 and self.conv2 is not None:
            return self._effective_weight(self.conv2)
        raise ValueError(f"expected layer 1 or 2, got {layer}")

    @property
    def weight(self):
        weights = [self.conv.weight.flatten()]
        if self.conv2 is not None:
            weights.append(self.conv2.weight.flatten())
        return torch.cat(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(
            x,
            self.effective_weight(1),
            self.conv.bias,
            padding=self.kernel // 2,
            groups=self.groups,
        )
        if self.conv2 is None:
            return y
        y = self.activation(y)
        return F.conv1d(
            y,
            self.effective_weight(2),
            self.conv2.bias,
            padding=self.kernel // 2,
            groups=self.groups,
        )

class MultiScaleSymmetricConv1d(nn.Module):
    """Per-channel zero-phase temporal filter bank with multiple kernel scales."""

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernels=(7, 15, 31, 61),
        conv_layers: int = 1,
    ):
        super().__init__()
        kernels = _parse_kernels(kernels)
        conv_layers = int(conv_layers)
        if conv_layers not in {1, 2}:
            raise ValueError("multiscale symmetric conv_layers must be 1 or 2")
        branch_dims = _split_dims(filters_per_channel, len(kernels))
        self.temporal_branches = nn.ModuleList(
            [
                SymmetricBranchConv1d(
                    in_channels,
                    branch_dim,
                    kernel,
                    conv_layers=conv_layers,
                )
                for kernel, branch_dim in zip(kernels, branch_dims)
            ]
        )
        self.in_channels = int(in_channels)
        self.filters_per_channel = int(filters_per_channel)
        self.kernels = kernels
        self.conv_layers = int(conv_layers)
        self.out_channels = self.in_channels * self.filters_per_channel

    @property
    def weight(self):
        return torch.cat([branch.weight for branch in self.temporal_branches])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        if N != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {N}")
        return torch.cat([branch(x) for branch in self.temporal_branches], dim=1)


class MultiScaleAntiSymmetricConv1d(nn.Module):
    """Per-channel derivative-like temporal filter bank with multiple kernel scales."""

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernels=(7, 15, 31, 61),
        conv_layers: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        kernels = _parse_kernels(kernels)
        conv_layers = int(conv_layers)
        if conv_layers not in {1, 2}:
            raise ValueError("multiscale anti-symmetric conv_layers must be 1 or 2")
        branch_dims = _split_dims(filters_per_channel, len(kernels))
        self.temporal_branches = nn.ModuleList(
            [
                AntiSymmetricBranchConv1d(
                    in_channels,
                    branch_dim,
                    kernel,
                    conv_layers=conv_layers,
                    bias=bias,
                )
                for kernel, branch_dim in zip(kernels, branch_dims)
            ]
        )
        self.in_channels = int(in_channels)
        self.filters_per_channel = int(filters_per_channel)
        self.kernels = kernels
        self.conv_layers = int(conv_layers)
        self.bias = bool(bias)
        self.out_channels = self.in_channels * self.filters_per_channel

    @property
    def weight(self):
        return torch.cat([branch.weight for branch in self.temporal_branches])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        if N != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {N}")
        return torch.cat([branch(x) for branch in self.temporal_branches], dim=1)


class MixedParityTemporalConv1d(nn.Module):
    """Independent symmetric and anti-symmetric temporal filter banks."""

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernels=(7, 15, 31, 61),
        conv_layers: int = 1,
    ):
        super().__init__()
        self.sym_conv = MultiScaleSymmetricConv1d(
            in_channels,
            filters_per_channel,
            kernels=kernels,
            conv_layers=conv_layers,
        )
        self.anti_conv = MultiScaleAntiSymmetricConv1d(
            in_channels,
            filters_per_channel,
            kernels=kernels,
            conv_layers=conv_layers,
            bias=False,
        )
        self.in_channels = int(in_channels)
        self.filters_per_channel = int(filters_per_channel)
        self.kernels = _parse_kernels(kernels)
        self.conv_layers = int(conv_layers)
        self.out_channels = self.sym_conv.out_channels + self.anti_conv.out_channels

    @property
    def weight(self):
        return torch.cat([self.sym_conv.weight, self.anti_conv.weight])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sym_conv(x), self.anti_conv(x)


# class ResidualBranch(nn.Module):
#     """Kernel-specific temporal branch from the CoCoT-style EEG embedder."""

#     def __init__(self, kernel: int, branch_dim: int):
#         super().__init__()
#         if kernel < 1 or kernel % 2 == 0:
#             raise ValueError("residual branch kernels must be positive odd integers")
#         padding = kernel // 2
#         groups = math.gcd(4, branch_dim)
#         self.conv = nn.Conv1d(1, branch_dim, kernel_size=kernel, padding=padding, bias=False)
#         self.norm = nn.GroupNorm(groups, branch_dim)
#         self.act = nn.GELU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.act(self.norm(self.conv(x)))


# class MultiScaleResidualConv1d(nn.Module):
#     """Per-channel multi-kernel residual temporal front-end.

#     Each EEG channel is reshaped to its own 1D sequence, matching the reference
#     repo's `ResidualBranch` usage before channel/time token mixing. Branch
#     outputs are concatenated and returned as temporal features only.
#     """

#     def __init__(self, in_channels: int, filters_per_channel: int, kernels=(3, 7, 15, 31)):
#         super().__init__()
#         kernels = _parse_kernels(kernels)
#         branch_dims = _split_dims(filters_per_channel, len(kernels))
#         self.temporal_branches = nn.ModuleList(
#             [
#                 ResidualBranch(kernel, branch_dim)
#                 for kernel, branch_dim in zip(kernels, branch_dims)
#             ]
#         )
#         self.in_channels = int(in_channels)
#         self.filters_per_channel = int(filters_per_channel)
#         self.kernels = kernels
#         self.out_channels = self.in_channels * self.filters_per_channel

#     @property
#     def weight(self):
#         return torch.cat([branch.conv.weight.flatten() for branch in self.temporal_branches])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, T = x.shape
#         if N != self.in_channels:
#             raise ValueError(f"expected {self.in_channels} channels, got {N}")
#         sequence = x.reshape(B * N, 1, T)
#         y = torch.cat([branch(sequence) for branch in self.temporal_branches], dim=1)
#         return y.reshape(B, N * self.filters_per_channel, T)


class MLP(nn.Module):
    """Per-timepoint MLP embedder with shared weights across time.

    Input  x : (B, N, T)  — batch size, N channels, T timesteps
    Output   : (B, d, T)  — embedded trajectories

    """

    def __init__(self, in_channels: int, d: int = 128, hidden_dim: int = 256, depth: int = 3, dropout: float = 0.0,
                 temporal_filters: int = 0, temporal_kernel_size: int = 31,
                 temporal_frontend: str = "symmetric", residual_kernels=(3, 7, 15, 31),
                 multiscale_symmetric_conv_layers: int = 1, antisymmetric_planes: int = 0):
        super().__init__()
        assert depth >= 1, "depth must be at least 1"

        temporal_frontend = (temporal_frontend or "symmetric").lower()
        self.temporal_frontend = temporal_frontend
        self.mixed_parity = temporal_frontend in {
            "mixed_parity",
            "mixed_symmetric_antisymmetric",
            "mixed_sym_anti",
            "sym_anti",
        }
        self.d = int(d)

        if self.mixed_parity:
            if temporal_filters <= 0:
                raise ValueError("mixed_parity requires temporal_filters > 0")
            if d % 2 != 0:
                raise ValueError(f"mixed_parity requires an even embedding dimension, got d={d}")
            n_planes = d // 2
            antisymmetric_planes = int(antisymmetric_planes)
            if antisymmetric_planes < 0:
                antisymmetric_planes = max(1, n_planes // 2)
            if antisymmetric_planes < 0 or antisymmetric_planes > n_planes:
                raise ValueError(
                    f"antisymmetric_planes must be between 0 and {n_planes}, "
                    f"got {antisymmetric_planes}"
                )
            self.antisymmetric_planes = antisymmetric_planes
            self.symmetric_planes = n_planes - antisymmetric_planes
            self.sym_out_dim = 2 * self.symmetric_planes
            self.anti_out_dim = 2 * self.antisymmetric_planes
            self.temporal_conv = MixedParityTemporalConv1d(
                in_channels,
                temporal_filters,
                kernels=residual_kernels,
                conv_layers=multiscale_symmetric_conv_layers,
            )
            conv_out_channels = self.temporal_conv.sym_conv.out_channels
            self.sym_net = (
                _make_pointwise_mlp(conv_out_channels, self.sym_out_dim, hidden_dim, depth, dropout)
                if self.sym_out_dim > 0
                else None
            )


            self.anti_net = _make_odd_pointwise_mlp(
                conv_out_channels,
                self.anti_out_dim,
                hidden_dim,
                depth,
            )
            self.net = None
            self._init_weights()
            return

        if temporal_filters > 0:
            if temporal_frontend in {"symmetric"}:
                # per-channel filter bank: each input channel -> temporal_filters filters
                self.temporal_conv = SymmetricConv1d(in_channels, temporal_filters, temporal_kernel_size)
            elif temporal_frontend in {"multiscale_symmetric", "symmetric_multiscale"}:
                self.temporal_conv = MultiScaleSymmetricConv1d(
                    in_channels,
                    temporal_filters,
                    kernels=residual_kernels,
                    conv_layers=multiscale_symmetric_conv_layers,
                )
            elif temporal_frontend in {"multiscale_antisymmetric", "antisymmetric_multiscale"}:
                self.temporal_conv = MultiScaleAntiSymmetricConv1d(
                    in_channels,
                    temporal_filters,
                    kernels=residual_kernels,
                    conv_layers=multiscale_symmetric_conv_layers,
                )
            elif temporal_frontend in {"residual"}:
                self.temporal_conv = MultiScaleResidualConv1d(
                    in_channels, temporal_filters, kernels=residual_kernels
                )
            else:
                raise ValueError(
                    "temporal_frontend must be one of: symmetric, multiscale_symmetric, "
                    "multiscale_antisymmetric, mixed_parity, residual"
                )
            in_channels = self.temporal_conv.out_channels   # use temporal features only (no raw concat)
        else:
            self.temporal_conv = None

        self.net = _make_pointwise_mlp(in_channels, d, hidden_dim, depth, dropout)
        self.antisymmetric_planes = 0
        self.symmetric_planes = d // 2 if d % 2 == 0 else 0
        self.sym_net = None
        self.anti_net = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        if self.mixed_parity:
            x_sym, x_anti = self.temporal_conv(x)
            parts = []
            if self.sym_net is not None:
                parts.append(_apply_pointwise_net(x_sym, self.sym_net, self.sym_out_dim))
            if self.anti_net is not None:
                parts.append(_apply_pointwise_net(x_anti, self.anti_net, self.anti_out_dim))
            return torch.cat(parts, dim=1)
        if self.temporal_conv is not None:
            # x = torch.cat([x, self.temporal_conv(x)], dim=1)   # (B, N + temporal_filters, T)
            x = self.temporal_conv(x)
        return _apply_pointwise_net(x, self.net, self.d)

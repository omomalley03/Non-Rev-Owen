import math
from collections import OrderedDict
from typing import Optional

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
    if dropout < 0.0 or dropout > 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")
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


def _apply_paired_sequential(
    a: torch.Tensor,
    b: torch.Tensor,
    net: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a pointwise network twice, sharing dropout masks per layer."""
    modules = net if isinstance(net, nn.Sequential) else nn.Sequential(net)
    for module in modules:
        if isinstance(module, nn.Dropout):
            p = float(module.p)
            if not module.training or p == 0.0:
                continue
            if p == 1.0:
                mask = torch.zeros_like(a)
            else:
                keep_probability = 1.0 - p
                mask = torch.empty_like(a).bernoulli_(keep_probability).div_(keep_probability)
            a = a * mask
            b = b * mask
            continue
        a = module(a)
        b = module(b)
    return a, b


def _apply_pointwise_net_paired(
    x: torch.Tensor,
    qx: torch.Tensor,
    net: nn.Module,
    out_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, T = x.shape
    y = x.permute(0, 2, 1).reshape(B * T, C)
    qy = qx.permute(0, 2, 1).reshape(B * T, C)
    y, qy = _apply_paired_sequential(y, qy, net)
    y = y.reshape(B, T, out_dim).permute(0, 2, 1)
    qy = qy.reshape(B, T, out_dim).permute(0, 2, 1)
    return y, qy


class ParityProjectedPointwiseHead(nn.Module):
    """Parity-projected pointwise MLP head for mixed temporal features.

    ``even_features`` are symmetric/even temporal features and ``odd_features``
    are antisymmetric/odd temporal features. The head concatenates
    ``u = (e, o)`` and applies the input parity transform

        Q(e, o) = (e, -o)

    to enforce output parity around otherwise arbitrary pointwise MLPs:

        f_+(u) = [g_+(u) + g_+(Qu)] / 2
        f_-(u) = [g_-(u) - g_-(Qu)] / 2

        f_+(Qu) =  f_+(u)
        f_-(Qu) = -f_-(u)

    Biases, affine LayerNorm, GELU, and dropout are allowed inside ``g_+`` and
    ``g_-`` because parity is imposed by the final projection. Dropout masks are
    coupled between the ``u`` and ``Qu`` evaluations of each network.
    """

    def __init__(
        self,
        even_in_dim: int,
        odd_in_dim: int,
        even_out_dim: int,
        odd_out_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        even_in_dim = int(even_in_dim)
        odd_in_dim = int(odd_in_dim)
        even_out_dim = int(even_out_dim)
        odd_out_dim = int(odd_out_dim)
        if even_in_dim < 0 or odd_in_dim < 0:
            raise ValueError("input dimensions must be nonnegative")
        if even_out_dim < 0 or odd_out_dim < 0:
            raise ValueError("output dimensions must be nonnegative")
        if even_out_dim == 0 and odd_out_dim == 0:
            raise ValueError("at least one projected output group must be nonempty")

        full_in_dim = even_in_dim + odd_in_dim
        if full_in_dim <= 0:
            raise ValueError("at least one input feature group must be nonempty")

        self.even_in_dim = even_in_dim
        self.odd_in_dim = odd_in_dim
        self.even_out_dim = even_out_dim
        self.odd_out_dim = odd_out_dim
        self.in_dim = full_in_dim

        self.even_net = (
            _make_pointwise_mlp(full_in_dim, even_out_dim, hidden_dim, depth, dropout)
            if even_out_dim > 0
            else None
        )
        self.odd_net = (
            _make_pointwise_mlp(full_in_dim, odd_out_dim, hidden_dim, depth, dropout)
            if odd_out_dim > 0
            else None
        )
        parity_sign = torch.cat([
            torch.ones(even_in_dim),
            -torch.ones(odd_in_dim),
        ])
        self.register_buffer("input_parity_sign", parity_sign.view(1, -1, 1))

    def _inputs(self, even_features: torch.Tensor, odd_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if even_features.shape[0] != odd_features.shape[0] or even_features.shape[-1] != odd_features.shape[-1]:
            raise ValueError("even and odd features must have matching batch and time dimensions")
        if even_features.shape[1] != self.even_in_dim:
            raise ValueError(f"expected {self.even_in_dim} even input features, got {even_features.shape[1]}")
        if odd_features.shape[1] != self.odd_in_dim:
            raise ValueError(f"expected {self.odd_in_dim} odd input features, got {odd_features.shape[1]}")
        u = torch.cat([even_features, odd_features], dim=1)
        return u, u * self.input_parity_sign

    def forward_parts(
        self,
        even_features: torch.Tensor,
        odd_features: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return projected even and odd outputs before final concatenation."""
        u, qu = self._inputs(even_features, odd_features)
        even_output = None
        odd_output = None
        if self.even_net is not None:
            y, qy = _apply_pointwise_net_paired(u, qu, self.even_net, self.even_out_dim)
            even_output = 0.5 * (y + qy)
        if self.odd_net is not None:
            y, qy = _apply_pointwise_net_paired(u, qu, self.odd_net, self.odd_out_dim)
            odd_output = 0.5 * (y - qy)
        return even_output, odd_output

    def forward_parts_with_parity_flip(
        self,
        even_features: torch.Tensor,
        odd_features: torch.Tensor,
    ) -> tuple[
        tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ]:
        """Return projected outputs for ``(e, o)`` and ``Q(e, o)`` using one dropout draw."""
        u, qu = self._inputs(even_features, odd_features)
        even_output = None
        even_flipped_output = None
        odd_output = None
        odd_flipped_output = None
        if self.even_net is not None:
            y, qy = _apply_pointwise_net_paired(u, qu, self.even_net, self.even_out_dim)
            even_output = 0.5 * (y + qy)
            even_flipped_output = 0.5 * (qy + y)
        if self.odd_net is not None:
            y, qy = _apply_pointwise_net_paired(u, qu, self.odd_net, self.odd_out_dim)
            odd_output = 0.5 * (y - qy)
            odd_flipped_output = 0.5 * (qy - y)
        return (even_output, odd_output), (even_flipped_output, odd_flipped_output)

    def forward(self, even_features: torch.Tensor, odd_features: torch.Tensor) -> torch.Tensor:
        parts = [part for part in self.forward_parts(even_features, odd_features) if part is not None]
        return torch.cat(parts, dim=1)


def _conv1d_with_trial_context(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    kernel_size: int,
    groups: int,
    context_bins: int,
) -> torch.Tensor:
    """Same-length conv using real trial context when it is present in ``x``."""
    radius = kernel_size // 2
    context_bins = int(context_bins)
    if context_bins <= 0:
        return F.conv1d(x, weight, bias, padding=radius, groups=groups)
    if context_bins < radius:
        raise ValueError(
            f"temporal_context_bins={context_bins} is too small for kernel_size={kernel_size}; "
            f"need at least {radius}"
        )
    if x.shape[-1] <= 2 * context_bins:
        raise ValueError(
            f"input length {x.shape[-1]} is too short for temporal_context_bins={context_bins}"
        )

    start = context_bins - radius
    stop = x.shape[-1] - context_bins + radius
    return F.conv1d(x[..., start:stop], weight, bias, padding=0, groups=groups)


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

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernel_size: int,
        context_bins: int = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "use an odd kernel for an exact zero-phase 'same' conv"
        out_channels = in_channels * filters_per_channel
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.uniform_(self.weight, -1.0 / kernel_size ** 0.5, 1.0 / kernel_size ** 0.5)
        self.padding = kernel_size // 2
        self.context_bins = int(context_bins)
        self.groups = in_channels
        self.out_channels = out_channels
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, in, T)
        w = self.weight + self.weight.flip(-1)            # palindromic in time
        y = _conv1d_with_trial_context(
            x, w, self.bias, self.padding * 2 + 1, self.groups, self.context_bins
        )
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
        context_bins: int = 0,
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
        self.context_bins = int(context_bins)

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
        y = _conv1d_with_trial_context(
            x,
            self.effective_weight(1),
            self.conv.bias,
            self.kernel,
            self.groups,
            self.context_bins,
        )
        if self.conv2 is None:
            return y
        y = self.activation(y)
        if self.context_bins > 0:
            y = F.pad(y, (self.kernel // 2, self.kernel // 2), mode="replicate")
            return F.conv1d(y, self.effective_weight(2), self.conv2.bias, padding=0, groups=self.groups)
        return F.conv1d(y, self.effective_weight(2), self.conv2.bias, padding=self.kernel // 2, groups=self.groups)

class AntiSymmetricBranchConv1d(nn.Module):
    """One derivative-like depthwise branch for a single temporal scale."""

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernel_size: int,
        conv_layers: int = 1,
        bias: bool = False,
        context_bins: int = 0,
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
        self.context_bins = int(context_bins)

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
        y = _conv1d_with_trial_context(
            x,
            self.effective_weight(1),
            self.conv.bias,
            self.kernel,
            self.groups,
            self.context_bins,
        )
        if self.conv2 is None:
            return y
        y = self.activation(y)
        if self.context_bins > 0:
            y = F.pad(y, (self.kernel // 2, self.kernel // 2), mode="replicate")
            return F.conv1d(y, self.effective_weight(2), self.conv2.bias, padding=0, groups=self.groups)
        return F.conv1d(y, self.effective_weight(2), self.conv2.bias, padding=self.kernel // 2, groups=self.groups)

class MultiScaleSymmetricConv1d(nn.Module):
    """Per-channel zero-phase temporal filter bank with multiple kernel scales."""

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernels=(7, 15, 31, 61),
        conv_layers: int = 1,
        context_bins: int = 0,
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
                    context_bins=context_bins,
                )
                for kernel, branch_dim in zip(kernels, branch_dims)
            ]
        )
        self.in_channels = int(in_channels)
        self.filters_per_channel = int(filters_per_channel)
        self.kernels = kernels
        self.conv_layers = int(conv_layers)
        self.context_bins = int(context_bins)
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
        context_bins: int = 0,
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
                    context_bins=context_bins,
                )
                for kernel, branch_dim in zip(kernels, branch_dims)
            ]
        )
        self.in_channels = int(in_channels)
        self.filters_per_channel = int(filters_per_channel)
        self.kernels = kernels
        self.conv_layers = int(conv_layers)
        self.bias = bool(bias)
        self.context_bins = int(context_bins)
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
    """Symmetric/even and antisymmetric/odd temporal filter banks.

    The returned pair ``(e, o)`` satisfies exact time-reversal parity:
    ``e(Rx) = R e(x)`` and ``o(Rx) = -R o(x)``.
    """

    def __init__(
        self,
        in_channels: int,
        filters_per_channel: int,
        kernels=(7, 15, 31, 61),
        conv_layers: int = 1,
        context_bins: int = 0,
    ):
        super().__init__()
        self.sym_conv = MultiScaleSymmetricConv1d(
            in_channels,
            filters_per_channel,
            kernels=kernels,
            conv_layers=conv_layers,
            context_bins=context_bins,
        )
        self.anti_conv = MultiScaleAntiSymmetricConv1d(
            in_channels,
            filters_per_channel,
            kernels=kernels,
            # The mixed-parity head relies on known odd frontend parity before
            # projection. A stack OddConv -> GELU -> OddConv does not preserve
            # that known parity because GELU is not an odd pointwise function.
            # Keep exactly one unbiased temporally antisymmetric convolution.
            conv_layers=1,
            bias=False,
            context_bins=context_bins,
        )
        self.in_channels = int(in_channels)
        self.filters_per_channel = int(filters_per_channel)
        self.kernels = _parse_kernels(kernels)
        self.conv_layers = int(conv_layers)
        self.anti_conv_layers = 1
        self.context_bins = int(context_bins)
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
                 multiscale_symmetric_conv_layers: int = 1, antisymmetric_planes: int = 0,
                 temporal_context_bins: int = 0):
        super().__init__()
        assert depth >= 1, "depth must be at least 1"

        temporal_frontend = (temporal_frontend or "symmetric").lower()
        self.temporal_frontend = temporal_frontend
        self.temporal_context_bins = int(temporal_context_bins)
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
            # These names are retained for configuration/checkpoint familiarity.
            # They now denote odd-output and even-output 2D planes respectively.
            self.antisymmetric_planes = antisymmetric_planes
            self.symmetric_planes = n_planes - antisymmetric_planes
            self.sym_out_dim = 2 * self.symmetric_planes
            self.anti_out_dim = 2 * self.antisymmetric_planes
            self.temporal_conv = MixedParityTemporalConv1d(
                in_channels,
                temporal_filters,
                kernels=residual_kernels,
                conv_layers=multiscale_symmetric_conv_layers,
                context_bins=self.temporal_context_bins,
            )
            self.parity_head = ParityProjectedPointwiseHead(
                self.temporal_conv.sym_conv.out_channels,
                self.temporal_conv.anti_conv.out_channels,
                self.sym_out_dim,
                self.anti_out_dim,
                hidden_dim,
                depth,
                dropout,
            )
            self.net = None
            self._init_weights()
            return

        if temporal_filters > 0:
            if temporal_frontend in {"symmetric"}:
                # per-channel filter bank: each input channel -> temporal_filters filters
                self.temporal_conv = SymmetricConv1d(
                    in_channels, temporal_filters, temporal_kernel_size,
                    context_bins=self.temporal_context_bins,
                )
            elif temporal_frontend in {"multiscale_symmetric", "symmetric_multiscale"}:
                self.temporal_conv = MultiScaleSymmetricConv1d(
                    in_channels,
                    temporal_filters,
                    kernels=residual_kernels,
                    conv_layers=multiscale_symmetric_conv_layers,
                    context_bins=self.temporal_context_bins,
                )
            elif temporal_frontend in {"multiscale_antisymmetric", "antisymmetric_multiscale"}:
                self.temporal_conv = MultiScaleAntiSymmetricConv1d(
                    in_channels,
                    temporal_filters,
                    kernels=residual_kernels,
                    conv_layers=multiscale_symmetric_conv_layers,
                    context_bins=self.temporal_context_bins,
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
        self.parity_head = None
        self._init_weights()

    @property
    def sym_net(self) -> Optional[nn.Module]:
        """Projected even-output MLP for mixed parity models, otherwise ``None``."""
        if self.parity_head is None:
            return None
        return self.parity_head.even_net

    @property
    def anti_net(self) -> Optional[nn.Module]:
        """Projected odd-output MLP for mixed parity models, otherwise ``None``."""
        if self.parity_head is None:
            return None
        return self.parity_head.odd_net

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.mixed_parity and self.parity_head is not None:
            state_dict = OrderedDict(state_dict)
            own_state = self.state_dict()
            translations = (
                ("sym_net.", "parity_head.even_net."),
                ("anti_net.", "parity_head.odd_net."),
            )
            for old_prefix, new_prefix in translations:
                for key, value in list(state_dict.items()):
                    if not key.startswith(old_prefix):
                        continue
                    new_key = new_prefix + key[len(old_prefix):]
                    if (
                        new_key in own_state
                        and new_key not in state_dict
                        and tuple(own_state[new_key].shape) == tuple(value.shape)
                    ):
                        state_dict[new_key] = value
                        del state_dict[key]
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

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
            return self.parity_head(x_sym, x_anti)
        if self.temporal_conv is not None:
            # x = torch.cat([x, self.temporal_conv(x)], dim=1)   # (B, N + temporal_filters, T)
            x = self.temporal_conv(x)
        return _apply_pointwise_net(x, self.net, self.d)

    def check_time_reversal_parity(self, x: torch.Tensor) -> dict[str, float]:
        """Return max absolute time-reversal parity errors for mixed-parity models."""
        if not self.mixed_parity:
            raise ValueError("time-reversal parity diagnostics are only defined for mixed_parity models")

        was_training = self.training
        self.eval()
        with torch.no_grad():
            e, o = self.temporal_conv(x)
            e_rev, o_rev = self.temporal_conv(x.flip(-1))
            y = self(x)
            y_rev = self(x.flip(-1))
            y_even = y[:, : self.sym_out_dim]
            y_odd = y[:, self.sym_out_dim :]
            y_rev_even = y_rev[:, : self.sym_out_dim]
            y_rev_odd = y_rev[:, self.sym_out_dim :]

            def max_abs(t: torch.Tensor) -> float:
                return float(t.abs().max().item()) if t.numel() > 0 else 0.0

            errors = {
                "symmetric_frontend_error": max_abs(e_rev - e.flip(-1)),
                "antisymmetric_frontend_error": max_abs(o_rev + o.flip(-1)),
                "even_output_error": max_abs(y_rev_even - y_even.flip(-1)),
                "odd_output_error": max_abs(y_rev_odd + y_odd.flip(-1)),
            }
        if was_training:
            self.train()
        return errors

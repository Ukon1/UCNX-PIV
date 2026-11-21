"""Original code of this work: the ConvNeXt block used by the encoder.

This file contains only the components introduced by the authors of this
repository.  It is released for non-commercial research and educational use
only; see LICENSE.

The block was designed with reference to ConvNeXt (Liu et al., CVPR 2022,
MIT license); the implementation here is the authors' own and is the block
that forms the encoder of ConvNeXt-PIV.  See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ChannelLayerNorm", "ConvNeXtBlock"]


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a [B, C, H, W] tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNeXtBlock(nn.Module):
    """Depth-wise 7x7 conv -> LN -> 4x MLP -> LayerScale -> residual.

    Original contribution of this work; this block is the core of the
    ConvNeXt-PIV encoder.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = ChannelLayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)
        return shortcut + x

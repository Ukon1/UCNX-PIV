"""Assembly of the UCNX-PIV network.

This file deliberately contains only the model assembly.  The code is split by
origin so that the provenance is unambiguous:

* ``convnext_original.py`` -- the original ``ConvNeXtBlock`` of this work
  (research-use-only; see LICENSE).
* ``raft_piv.py`` -- residual down-sampling, correlation pyramid, ConvGRU
  update block and convex upsampling, following RAFT-PIV/RAFT
  (BSD 3-Clause; see THIRD_PARTY_NOTICES.md).
* ``model.py`` (this file) -- the encoder assembly and the top-level
  ``PIVNet`` that combine the two.

The module names and parameter shapes are kept identical to the training
code so that the released checkpoint loads with ``strict=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from convnext_original import ConvNeXtBlock
from raft_piv import (
    BasicUpdateBlock256,
    CorrBlock,
    ResidualBlock,
    convex_upsample_flow,
    coords_grid,
)

__all__ = ["PIVNet", "ConvNeXtEncoder256"]


class ConvNeXtEncoder256(nn.Module):
    """Particle-image encoder producing features at 1/4 resolution.

    The stage blocks are the original ``ConvNeXtBlock``.  The stem and the
    residual down-sampling blocks follow the RAFT-PIV encoder layout.  Input
    is either a single [B, 1, H, W] tensor or a pair of tensors; output has
    ``output_dim`` channels at [B, C, H/4, W/4].
    """

    def __init__(
        self,
        output_dim: int = 128,
        in_chans: int = 1,
        dims: tuple[int, int] = (64, 96),
        depths: tuple[int, int] = (3, 3),
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.stem_conv = nn.Conv2d(
            in_chans, 32, kernel_size=7, stride=1, padding=3
        )
        self.stem_norm = nn.InstanceNorm2d(32)
        self.stem_act = nn.ReLU(inplace=True)

        self.res_down1 = ResidualBlock(32, dims[0], stride=2)
        self.stage1 = nn.Sequential(
            *[
                ConvNeXtBlock(dims[0], mlp_ratio, layer_scale_init_value)
                for _ in range(depths[0])
            ]
        )

        self.res_down2 = ResidualBlock(dims[0], dims[1], stride=2)
        self.stage2 = nn.Sequential(
            *[
                ConvNeXtBlock(dims[1], mlp_ratio, layer_scale_init_value)
                for _ in range(depths[1])
            ]
        )

        self.head = nn.Conv2d(dims[1], output_dim, kernel_size=1)

    def forward(self, x):
        packed = isinstance(x, (list, tuple))
        if packed:
            batch = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.res_down1(x)
        x = self.stage1(x)
        x = self.res_down2(x)
        x = self.stage2(x)
        x = self.head(x)

        if packed:
            x = torch.split(x, [batch, batch], dim=0)
        return x


class PIVNet(nn.Module):
    """UCNX-PIV: shared ConvNeXt encoder + RAFT-PIV recurrent refinement."""

    def __init__(self, iters: int = 16):
        super().__init__()
        self.iters = iters
        self.fnet = ConvNeXtEncoder256(
            output_dim=128,
            in_chans=1,
            dims=(64, 96),
            depths=(3, 3),
        )
        self.cnet = ConvNeXtEncoder256(
            output_dim=128,
            in_chans=1,
            dims=(64, 96),
            depths=(3, 3),
        )
        self.update_block = BasicUpdateBlock256(
            corr_levels=4, corr_radius=4, hidden_dim=64
        )

    @staticmethod
    def _initial_coords(img: torch.Tensor):
        n, _, h, w = img.shape
        coords = coords_grid(n, h // 4, w // 4, img.device, img.dtype)
        return coords.clone(), coords.clone()

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        iters: int | None = None,
    ) -> torch.Tensor:
        """Estimate the displacement field from ``img1`` to ``img2``.

        Args:
            img1, img2: [B, 1, H, W] grayscale images in [0, 1].
            iters: number of recurrent refinement steps (default: self.iters).

        Returns:
            [B, 2, H, W] displacement field in pixels.
        """
        if img1.shape[-2:] != img2.shape[-2:]:
            raise ValueError("img1 and img2 must have the same spatial size")
        if img1.shape[-2] % 4 or img1.shape[-1] % 4:
            raise ValueError("H and W must be divisible by 4")

        steps = self.iters if iters is None else int(iters)
        if steps < 1:
            raise ValueError("iters must be at least 1")

        fmap1, fmap2 = self.fnet([img1, img2])
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=4, radius=4)

        context = self.cnet(img1)
        net, inp = torch.split(context, [64, 64], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self._initial_coords(img1)
        mask = None
        for _ in range(steps):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            net, mask, delta = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta

        return convex_upsample_flow(coords1 - coords0, mask)

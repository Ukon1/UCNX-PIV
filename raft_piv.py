"""Third-party-derived network components.

Everything in this file follows RAFT-PIV (Lagemann et al., Nature Machine
Intelligence 2021) and its upstream RAFT (Teed & Deng, ECCV 2020):

* the residual down-sampling block used between encoder stages;
* the all-pairs correlation pyramid (4 levels, radius 4);
* the motion-feature encoder, separable ConvGRU and flow head;
* convex (9x9) upsampling of the 1/4-resolution flow field.

These components are not claimed as original work of the authors of this
repository.  RAFT is released under the BSD 3-Clause license and the full
notice is reproduced in THIRD_PARTY_NOTICES.md.  The implementation was
rewritten for this release against the published architecture and the
released checkpoint's parameter layout; no upstream source file is copied
verbatim into this repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResidualBlock",
    "coords_grid",
    "bilinear_sampler",
    "CorrBlock",
    "FlowHead",
    "SepConvGRU",
    "BasicMotionEncoder",
    "BasicUpdateBlock256",
    "convex_upsample_flow",
]


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a strided 1x1 shortcut (RAFT-PIV style)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.InstanceNorm2d(planes)
        self.norm2 = nn.InstanceNorm2d(planes)
        if stride != 1:
            self.norm3 = nn.InstanceNorm2d(planes)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


def coords_grid(batch: int, height: int, width: int, device, dtype):
    """Pixel-centre coordinates as [B, 2, H, W], in (x, y) order."""
    y = torch.arange(height, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(width, device=device, dtype=dtype).view(1, -1)
    y = y.expand(height, width)
    x = x.expand(height, width)
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def bilinear_sampler(image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample ``image`` at pixel coordinates ``coords`` (x, y order)."""
    height, width = image.shape[-2:]
    grid_x, grid_y = coords.split([1, 1], dim=-1)
    grid_x = 2.0 * grid_x / (width - 1) - 1.0
    grid_y = 2.0 * grid_y / (height - 1) - 1.0
    grid = torch.cat([grid_x, grid_y], dim=-1).to(image.dtype)
    return F.grid_sample(image, grid, align_corners=True)


class CorrBlock:
    """Multi-scale all-pairs correlation lookup (RAFT)."""

    def __init__(self, fmap1, fmap2, num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius
        self.pyramid = []

        batch, _, h, w = fmap1.shape
        corr = torch.matmul(
            fmap1.reshape(batch, fmap1.shape[1], h * w).transpose(1, 2),
            fmap2.reshape(batch, fmap2.shape[1], h * w),
        )
        corr = corr.reshape(batch, h, w, 1, h, w)
        corr = corr / torch.sqrt(torch.tensor(fmap1.shape[1], dtype=torch.float32))
        corr = corr.reshape(batch * h * w, 1, h, w)

        self.pyramid.append(corr)
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.pyramid.append(corr)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h, w, _ = coords.shape

        span = torch.linspace(-r, r, 2 * r + 1)
        dy = span.view(-1, 1).expand(2 * r + 1, 2 * r + 1)
        dx = span.view(1, -1).expand(2 * r + 1, 2 * r + 1)
        # Keep the upstream channel order: stack (dy, dx) along the last axis.
        delta = torch.stack((dy, dx), dim=-1).to(coords.device)

        levels = []
        for level in range(self.num_levels):
            corr = self.pyramid[level]
            centroid = coords.reshape(batch * h * w, 1, 1, 2) / (2 ** level)
            sample_coords = centroid + delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            sampled = bilinear_sampler(corr, sample_coords)
            levels.append(sampled.reshape(batch, h, w, -1))

        out = torch.cat(levels, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class FlowHead(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 64, input_dim: int = 128):
        super().__init__()
        in_ch = hidden_dim + input_dim
        self.convz1 = nn.Conv2d(in_ch, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(in_ch, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(in_ch, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(in_ch, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(in_ch, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(in_ch, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels: int = 4, corr_radius: int = 4):
        super().__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 64, 1)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 32, 7, padding=3)
        self.convf2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv = nn.Conv2d(64 + 32, 64 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock256(nn.Module):
    def __init__(self, corr_levels: int = 4, corr_radius: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 16 * 9, 1),
        )

    def forward(self, net, inp, corr, flow):
        motion = self.encoder(flow, corr)
        net = self.gru(net, torch.cat([inp, motion], dim=1))
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


def convex_upsample_flow(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upsample a 1/4-resolution flow field by 4x with a 9x9 convex mask."""
    n, _, h, w = flow.shape
    mask = torch.softmax(mask.view(n, 1, 9, 4, 4, h, w), dim=2)
    up_flow = F.unfold(4 * flow, (3, 3), padding=1).view(n, 2, 9, 1, 1, h, w)
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(n, 2, 4 * h, 4 * w)

"""Image and optical-flow input/output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "read_gray",
    "load_image_pair",
    "prepare_for_model",
    "resize_flow",
    "read_flo",
    "write_flo",
]

FLO_MAGIC = 202021.25


def read_gray(path: str | Path) -> np.ndarray:
    """Read a grayscale image as float32 in [0, 1].

    TIFF files are read with ``tifffile`` so that 16-bit PIV images keep their
    full precision.  Other formats fall back to Pillow.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        image = tifffile.imread(path)
    else:
        from PIL import Image

        image = np.asarray(Image.open(path))

    if image.ndim == 3:
        # Collapse accidental RGB channels; PIV images are grayscale.
        image = image[..., :3].mean(axis=-1)

    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.5:
            # Floating-point images stored on a 0-255 scale.
            image /= 255.0
    return image


def load_image_pair(
    img1_path: str | Path, img2_path: str | Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load two images as [1, 1, H, W] float tensors."""
    a = torch.from_numpy(read_gray(img1_path)).unsqueeze(0).unsqueeze(0)
    b = torch.from_numpy(read_gray(img2_path)).unsqueeze(0).unsqueeze(0)
    if a.shape != b.shape:
        raise ValueError(
            f"image sizes differ: {tuple(a.shape[-2:])} vs {tuple(b.shape[-2:])}"
        )
    return a, b


def prepare_for_model(
    img1: torch.Tensor, img2: torch.Tensor, size: int = 256
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int] | None]:
    """Resize a pair to the training resolution.

    Returns the resized pair and the original (H, W), or ``None`` when no
    resize was needed.
    """
    height, width = img1.shape[-2:]
    if (height, width) == (size, size):
        return img1, img2, None

    def _resize(t):
        return F.interpolate(
            t, size=(size, size), mode="bilinear", align_corners=False
        )

    return _resize(img1), _resize(img2), (height, width)


def resize_flow(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a pixel-displacement field to ``(height, width)``.

    Bilinear interpolation is used for the field itself; the two components
    are then multiplied by the per-axis scale so that displacements remain in
    pixel units of the target grid.
    """
    src_h, src_w = flow.shape[-2:]
    if (src_h, src_w) == (height, width):
        return flow

    out = F.interpolate(
        flow, size=(height, width), mode="bilinear", align_corners=False
    )
    out[:, 0] *= width / src_w
    out[:, 1] *= height / src_h
    return out


def read_flo(path: str | Path) -> np.ndarray:
    """Read a standard ``.flo`` file and return a [H, W, 2] float32 array."""
    path = Path(path)
    with path.open("rb") as handle:
        magic = np.fromfile(handle, np.float32, count=1)
        if magic.size != 1 or magic[0] != FLO_MAGIC:
            raise ValueError(f"not a .flo file (bad magic): {path}")
        width = int(np.fromfile(handle, np.int32, count=1)[0])
        height = int(np.fromfile(handle, np.int32, count=1)[0])
        data = np.fromfile(handle, np.float32, count=2 * width * height)
    if data.size != 2 * width * height:
        raise ValueError(f"truncated .flo file: {path}")
    return data.reshape(height, width, 2)


def write_flo(path: str | Path, flow: np.ndarray | torch.Tensor) -> None:
    """Write a [H, W, 2] or [2, H, W] displacement field to ``.flo``."""
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("flow must have shape [H, W, 2] or [2, H, W]")

    height, width = flow.shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.array([FLO_MAGIC], dtype=np.float32).tofile(handle)
        np.array([width], dtype=np.int32).tofile(handle)
        np.array([height], dtype=np.int32).tofile(handle)
        flow.astype(np.float32).tofile(handle)

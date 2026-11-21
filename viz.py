"""Visualisation helpers (no dependency on OpenCV or Matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["flow_to_rgb", "save_rgb", "save_gray"]


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV -> RGB conversion, all inputs in [0, 1]."""
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def flow_to_rgb(
    flow: np.ndarray,
    max_magnitude: float | None = None,
    percentile: float = 99.5,
) -> np.ndarray:
    """Colour-code a [H, W, 2] displacement field.

    Hue encodes direction, saturation encodes displacement magnitude.  When
    ``max_magnitude`` is None the given percentile of the magnitude field is
    used, so a few outliers do not wash out the colour scale.
    """
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)

    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u * u + v * v)

    if max_magnitude is None:
        max_magnitude = float(np.percentile(magnitude, percentile))
    max_magnitude = max(float(max_magnitude), 1e-6)

    hue = (np.arctan2(v, u) / (2.0 * np.pi)) % 1.0
    saturation = np.clip(magnitude / max_magnitude, 0.0, 1.0)
    value = np.ones_like(magnitude)

    rgb = _hsv_to_rgb(hue, saturation, value)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def save_gray(path: str | Path, array: np.ndarray, vmin=None, vmax=None) -> None:
    """Save a scalar field as an 8-bit grayscale PNG with fixed scaling."""
    from PIL import Image

    array = np.asarray(array, dtype=np.float32)
    lo = float(array.min()) if vmin is None else float(vmin)
    hi = float(array.max()) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255.0 + 0.5).astype(np.uint8)).save(path)

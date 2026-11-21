"""Evaluation metrics for displacement fields."""

from __future__ import annotations

import numpy as np

__all__ = ["endpoint_error", "summarize"]


def _as_hw2(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("flow must have shape [H, W, 2] or [2, H, W]")
    return flow


def endpoint_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean endpoint error."""
    pred = _as_hw2(pred)
    target = _as_hw2(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    delta = pred - target
    return np.sqrt(np.sum(delta * delta, axis=-1))


def summarize(pred: np.ndarray, target: np.ndarray) -> dict:
    """Summary statistics used in the paper (EPE in pixels)."""
    pred = _as_hw2(pred)
    target = _as_hw2(target)
    epe = endpoint_error(pred, target)

    delta = pred - target
    rmse_u = float(np.sqrt(np.mean(delta[..., 0] ** 2)))
    rmse_v = float(np.sqrt(np.mean(delta[..., 1] ** 2)))

    return {
        "epe_mean": float(epe.mean()),
        "epe_median": float(np.median(epe)),
        "epe_p90": float(np.percentile(epe, 90)),
        "epe_max": float(epe.max()),
        "acc_1px": float(np.mean(epe < 1.0)),
        "acc_3px": float(np.mean(epe < 3.0)),
        "acc_5px": float(np.mean(epe < 5.0)),
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "n_pixels": int(epe.size),
    }

#!/usr/bin/env python
"""Run UCNX-PIV on a pair of PIV images.

Example
-------
python predict.py \
    --img1 examples/five_cases/backstep/backstep_img1.tif \
    --img2 examples/five_cases/backstep/backstep_img2.tif \
    --gt   examples/five_cases/backstep/backstep_flow.flo \
    --out  outputs/backstep
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from metrics import endpoint_error, summarize
from model import PIVNet
from pivio import load_image_pair, prepare_for_model, read_flo, resize_flow, write_flo
from viz import flow_to_rgb, save_gray, save_rgb

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a PIV displacement field with UCNX-PIV."
    )
    parser.add_argument("--img1", required=True, help="first particle image")
    parser.add_argument("--img2", required=True, help="second particle image")
    parser.add_argument("--gt", default=None, help="optional ground-truth .flo")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "weights" / "ucnx_piv.pth"),
        help="model checkpoint (default: weights/ucnx_piv.pth)",
    )
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--iters", type=int, default=16, help="refinement steps")
    parser.add_argument(
        "--size", type=int, default=256, help="network input size (training: 256)"
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="device"
    )
    parser.add_argument(
        "--amp", action="store_true", help="use float16 autocast on CUDA"
    )
    parser.add_argument(
        "--no-panel", action="store_true", help="do not write the comparison PNG"
    )
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(weights: str | Path, device: torch.device, iters: int) -> PIVNet:
    model = PIVNet(iters=iters).to(device)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def inference_context(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    model = load_model(args.weights, device, args.iters)

    img1, img2 = load_image_pair(args.img1, args.img2)
    img1_display = img1[0, 0].numpy()
    img1, img2, original_size = prepare_for_model(img1, img2, size=args.size)
    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.inference_mode(), inference_context(device, args.amp):
        pred = model(img1, img2, iters=args.iters)

    if original_size is not None:
        pred = resize_flow(pred, *original_size)
    pred = pred[0].float().cpu().numpy()  # [2, H, W]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.img1).stem.replace("_img1", "")

    flow_path = out_dir / f"{stem}_pred.flo"
    write_flo(flow_path, pred)
    np.save(out_dir / f"{stem}_pred.npy", pred)

    flow_hw2 = pred.transpose(1, 2, 0)
    save_rgb(out_dir / f"{stem}_pred_color.png", flow_to_rgb(flow_hw2))

    result = {"input": str(args.img1), "output": str(flow_path)}
    if args.gt:
        target = read_flo(args.gt)
        stats = summarize(flow_hw2, target)
        result.update(stats)
        save_rgb(out_dir / f"{stem}_gt_color.png", flow_to_rgb(target))
        save_gray(out_dir / f"{stem}_epe.png", endpoint_error(flow_hw2, target))

        if not args.no_panel:
            panels = [
                _to_uint8(img1_display),
                flow_to_rgb(flow_hw2),
                flow_to_rgb(target),
                _repeat_gray(endpoint_error(flow_hw2, target)),
            ]
            save_rgb(out_dir / f"{stem}_panel.png", np.hstack(panels))

    with (out_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"device       : {device}")
    print(f"flow written : {flow_path}")
    if args.gt:
        print(
            "EPE          : "
            f"mean {result['epe_mean']:.4f} px, "
            f"median {result['epe_median']:.4f} px, "
            f"p90 {result['epe_p90']:.4f} px"
        )
        print(
            "accuracy     : "
            f"<1 px {100 * result['acc_1px']:.2f}%, "
            f"<3 px {100 * result['acc_3px']:.2f}%, "
            f"<5 px {100 * result['acc_5px']:.2f}%"
        )


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        hi = lo + 1e-6
    gray = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    gray = (gray * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _repeat_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    hi = float(np.percentile(image, 99.5))
    hi = max(hi, 1e-6)
    gray = np.clip(image / hi, 0.0, 1.0)
    gray = (gray * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


if __name__ == "__main__":
    main()

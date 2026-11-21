#!/usr/bin/env python
"""Evaluate UCNX-PIV on a directory of PIV image pairs.

The script finds every ``*_img1.tif`` below ``--data-dir``, uses the matching
``*_img2.tif`` and ``*_flow.flo`` files, and writes one CSV row per pair plus
a mean/median summary.

Example
-------
python evaluate.py --data-dir examples/five_cases --out outputs/five_cases
python evaluate.py --data-dir /path/to/PIV_dataset/PIV-genImages/data \
    --split-file splits/test_seed1.txt --out outputs/test_split
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from pathlib import Path

import numpy as np
import torch

from metrics import summarize
from model import PIVNet
from pivio import load_image_pair, prepare_for_model, read_flo, resize_flow, write_flo

ROOT = Path(__file__).resolve().parent

METRIC_KEYS = [
    "epe_mean",
    "epe_median",
    "epe_p90",
    "epe_max",
    "acc_1px",
    "acc_3px",
    "acc_5px",
    "rmse_u",
    "rmse_v",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation of UCNX-PIV.")
    parser.add_argument("--data-dir", required=True, help="directory with *_img1.tif")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "weights" / "ucnx_piv.pth"),
        help="model checkpoint (default: weights/ucnx_piv.pth)",
    )
    parser.add_argument("--out", default="outputs/eval", help="output directory")
    parser.add_argument("--iters", type=int, default=16, help="refinement steps")
    parser.add_argument("--size", type=int, default=256, help="network input size")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="device"
    )
    parser.add_argument("--amp", action="store_true", help="float16 autocast on CUDA")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only N pairs")
    parser.add_argument(
        "--split-file",
        default=None,
        help="text file with one relative *_img1.tif path per line",
    )
    parser.add_argument(
        "--save-pred", action="store_true", help="also write predicted .flo files"
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


def collect_pairs(root: Path, split_file: str | None = None) -> list[Path]:
    if split_file:
        lines = [
            line.strip()
            for line in Path(split_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        files = [root / line for line in lines]
        missing = [path for path in files if not path.exists()]
        if missing:
            preview = "\n".join(str(path) for path in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} files from the split were not found, e.g.\n{preview}"
            )
        return files

    files = sorted(root.rglob("*_img1.tif"))
    if not files:
        raise FileNotFoundError(f"no *_img1.tif found below {root}")
    return files


def inference_context(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    root = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(root, args.split_file)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    model = load_model(args.weights, device, args.iters)

    rows: list[dict] = []
    for index, img1_path in enumerate(pairs, start=1):
        img2_path = img1_path.with_name(
            img1_path.name.replace("_img1.tif", "_img2.tif")
        )
        gt_path = img1_path.with_name(
            img1_path.name.replace("_img1.tif", "_flow.flo")
        )
        if not img2_path.exists():
            print(f"[skip] missing second image: {img2_path}")
            continue

        img1, img2 = load_image_pair(img1_path, img2_path)
        img1, img2, original_size = prepare_for_model(img1, img2, size=args.size)
        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.inference_mode(), inference_context(device, args.amp):
            pred = model(img1, img2, iters=args.iters)

        if original_size is not None:
            pred = resize_flow(pred, *original_size)
        pred = pred[0].float().cpu().numpy()

        name = img1_path.name.replace("_img1.tif", "")
        relative = img1_path.relative_to(root).parent.as_posix()

        row = {"sample": name, "subset": relative}
        if gt_path.exists():
            row.update(summarize(pred, read_flo(gt_path)))
        else:
            row.update({key: float("nan") for key in [*METRIC_KEYS, "n_pixels"]})
        rows.append(row)

        if args.save_pred:
            write_flo(out_dir / "pred" / f"{name}.flo", pred)

        epe = row["epe_mean"]
        print(f"[{index:>4}/{len(pairs)}] {name:<40} EPE {epe:.4f} px")

    if not rows:
        raise RuntimeError("no valid pairs were evaluated")

    csv_path = out_dir / "metrics.csv"
    fieldnames = ["sample", "subset", *METRIC_KEYS, "n_pixels"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in fieldnames} for row in rows
        )

    valid = [row for row in rows if np.isfinite(row["epe_mean"])]
    print(f"\n{len(rows)} pairs, {len(valid)} with ground truth")
    if valid:
        summary_rows = []
        case_means = []
        print(f"{'subset':<28}{'n':>5}{'EPE mean':>12}{'EPE median':>14}")
        for subset in sorted({row["subset"] for row in valid}):
            group = [row for row in valid if row["subset"] == subset]
            case_mean = float(np.mean([r["epe_mean"] for r in group]))
            case_median = float(np.median([r["epe_median"] for r in group]))
            case_means.append(case_mean)
            summary_rows.append(
                {
                    "subset": subset,
                    "n": len(group),
                    "epe_mean": case_mean,
                    "epe_median": case_median,
                }
            )
            print(
                f"{subset:<28}{len(group):>5}"
                f"{case_mean:>12.4f}{case_median:>14.4f}"
            )

        pooled = float(np.mean([r["epe_mean"] for r in valid]))
        pooled_median = float(np.median([r["epe_median"] for r in valid]))
        case_mean = float(np.mean(case_means))
        print(
            f"{'ALL (pooled)':<28}{len(valid):>5}"
            f"{pooled:>12.4f}{pooled_median:>14.4f}"
        )
        print(f"{'CASE MEAN (equal weight)':<28}{len(case_means):>5}{case_mean:>12.4f}")

        summary_rows.append(
            {"subset": "ALL (pooled)", "n": len(valid), "epe_mean": pooled, "epe_median": pooled_median}
        )
        summary_rows.append(
            {"subset": "CASE MEAN (equal weight)", "n": len(case_means), "epe_mean": case_mean, "epe_median": ""}
        )
        summary_path = out_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["subset", "n", "epe_mean", "epe_median"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary: {summary_path}")

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

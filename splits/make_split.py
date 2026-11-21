#!/usr/bin/env python
"""Regenerate the 10% held-out split used in the paper.

The split is built independently for each case from the alphabetically sorted
``*_img1.tif`` list, with a fixed torch generator seed of 1.  It matches the
split used by the training/evaluation code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import random_split

CASES = ["backstep", "cylinder", "JHTDB_channel", "DNS_turbulence", "SQG"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dataset root")
    parser.add_argument("--out", default="splits/test_seed1.txt")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.data_dir)
    lines = []
    for case in CASES:
        files = sorted((root / case).glob("*_img1.tif"))
        if not files:
            raise FileNotFoundError(f"no *_img1.tif in {root / case}")
        total = len(files)
        val_size = int(total * 0.1)
        train_size = total - val_size
        _, val_set = random_split(
            files,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        chosen = sorted(
            files[i].relative_to(root).as_posix() for i in val_set.indices
        )
        lines.extend(chosen)
        print(f"{case:<16} total {total:>5}  test {len(chosen):>4}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} pairs)")


if __name__ == "__main__":
    main()

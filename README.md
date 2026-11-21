# UCNX-PIV

Minimal inference release for *Teacher-Guided Trajectory Consistency Learning
for Unsupervised Particle Image Velocimetry* (Z. Xiao, B. Hong, L. Shan;
manuscript under review).  The repository contains the trained
UnConvNeXt-PIV / UCNX-PIV model, a self-contained inference and evaluation
implementation, and five sample image pairs for a quick code check.

The repository provides the implementation and trained weights for running
the released model.  The original code, trained weights, and documentation are
released for **non-commercial research and educational use only**; see
[LICENSE](LICENSE).

## Contents

```
.
├── convnext_original.py     # ORIGINAL ConvNeXtBlock (research-use-only)
├── raft_piv.py              # THIRD-PARTY RAFT-PIV/RAFT components (BSD-3)
├── model.py                 # assembly: encoder + PIVNet
├── predict.py               # single image pair -> displacement field
├── evaluate.py              # batch evaluation -> metrics.csv
├── pivio.py                 # TIF reading, resizing, .flo I/O
├── metrics.py               # endpoint error and summary statistics
├── viz.py                   # colour-coded flow visualisation
├── weights/ucnx_piv.pth     # released checkpoint
├── examples/
│   └── five_cases/          # five sample pairs for a quick code check
├── splits/
│   ├── test_seed1.txt       # fixed 10% held-out split
│   └── README.md            # how the split was generated
├── REPRODUCING.md           # benchmark evaluation commands
├── LICENSE
├── THIRD_PARTY_NOTICES.md
└── requirements.txt
```

## Installation

Python 3.10+ and PyTorch 2.x are recommended.  A CUDA GPU is optional; the code
also runs on CPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
# Install the PyTorch build that matches your CUDA driver from
# https://pytorch.org/get-started/locally/ if the default wheel is unsuitable.
```

No custom CUDA extension is required.

## Quick start

### One image pair

```bash
python predict.py \
  --img1 examples/five_cases/backstep/backstep_img1.tif \
  --img2 examples/five_cases/backstep/backstep_img2.tif \
  --gt   examples/five_cases/backstep/backstep_flow.flo \
  --out  outputs/backstep
```

This writes:

| file | content |
| --- | --- |
| `<name>_pred.flo` | predicted displacement field, standard Middlebury `.flo` |
| `<name>_pred.npy` | the same field as `float32 [2, H, W]` |
| `<name>_pred_color.png` | direction/magnitude colour map |
| `<name>_gt_color.png` | ground-truth colour map (when `--gt` is given) |
| `<name>_epe.png` | per-pixel endpoint error (when `--gt` is given) |
| `<name>_panel.png` | image / prediction / truth / EPE side by side |
| `<name>_metrics.json` | numeric metrics (when `--gt` is given) |

Use `--iters 16` (default) to match the paper.  `--amp` enables float16
autocast on CUDA.  `--device cpu` forces CPU inference.

### Quick check with the five sample pairs

```bash
python evaluate.py --data-dir examples/five_cases --out outputs/sample_check
```

The five pairs in `examples/five_cases/` are sample data for checking that
the code runs and the checkpoint loads.  The script writes `metrics.csv` and
`summary.csv`.  The benchmark evaluation in the paper uses the complete test
set; the exact split file and commands are given in
[REPRODUCING.md](REPRODUCING.md).

## Public datasets used in the paper

The paper uses the public datasets below for training and/or evaluation.

| dataset / case | what was used | evaluation | public source |
| --- | --- | --- | --- |
| Cai et al. PIV benchmark | five synthetic flow scenarios: backstep, cylinder, JHTDB channel, DNS turbulence, SQG; 256 x 256 images with ground truth | quantitative (AEE / EPE) | <https://github.com/shengzesnail/PIV_dataset>, doi:[10.1007/s00348-019-2717-2](https://doi.org/10.1007/s00348-019-2717-2) |
| JHTDB | turbulent channel velocity fields used to generate the JHTDB channel case | underlying flow fields | <http://turbulence.pha.jhu.edu/> |
| PIV Challenge jet flow | 100 consecutive experimental pairs, 512 x 512, no ground truth | qualitative | <http://www.pivchallenge.org/>, Westerweel & Scarano (2005), doi:[10.1007/s00348-005-0016-6](https://doi.org/10.1007/s00348-005-0016-6) |
| Vortex pair | experimental vortex pair, no ground truth | qualitative | images in <https://github.com/shengzesnail/PIV-LiteFlowNet-en/tree/master/demos/testedData>; experiment: Willert & Gharib (1997), doi:[10.1017/S0022112097006265](https://doi.org/10.1017/S0022112097006265) |

For the fixed 10% held-out split provided in `splits/test_seed1.txt`,
download the full Cai benchmark and run:

```bash
python evaluate.py \
  --data-dir /path/to/PIV_dataset/PIV-genImages/data \
  --split-file splits/test_seed1.txt \
  --out outputs/test_split
```

`evaluate.py` searches recursively for `*_img1.tif` and uses the matching
`*_img2.tif` and `*_flow.flo`, so no per-case configuration is needed.  The
per-case means and the equal-weight mean are written to `summary.csv`.

## Input and output conventions

* Images are read as grayscale.  `uint8` is divided by 255 and `uint16` by
  65535; floating-point images already in `[0, 1]` are left untouched.
* The network was trained at `256 x 256`.  Pairs of that size are passed
  through unchanged.  Other sizes are resized bilinearly to `256 x 256`, and
  the predicted displacement is resized back to the original grid with the
  two components scaled by the per-axis resize factor.
* Displacements are in **pixels** of the input grid, with `u` along the
  horizontal axis and `v` along the vertical axis.
* `.flo` files follow the standard Middlebury layout (`magic = 202021.25`,
  `width`, `height`, then `float32` `u, v` samples).

## Model and checkpoint

* `weights/ucnx_piv.pth` — the checkpoint used for the main results in the
  paper (UCNX-PIV, 16 refinement iterations, convex upsampling).  It is the
  weight file referred to as `1.pth` in the authors' experiment tree.
* 170 parameter tensors, about **1.70 M** parameters.
* MD5 `574692ad17bb5dfa449b974ca61499ce`
* SHA-256 `f26aa8274db855e9467a10c662e288ba5482874de483e4bac3a165d3e630ba10`

The checkpoint was saved with a `module.` prefix from data-parallel training;
the loader strips it automatically.  No other preprocessing of the state dict
is performed.

Network overview: a shared ConvNeXt-style encoder produces 1/4-resolution
features for both frames; an all-pairs correlation pyramid and a ConvGRU
update block refine the displacement for 16 iterations; a 9x9 convex mask
upsamples the 1/4-resolution field to full resolution.

## Statement of original code

The original code contribution of this work is the **`ConvNeXtBlock`** in
[`convnext_original.py`](convnext_original.py): the 7x7 depth-wise
convolution, channel-last LayerNorm, 4x inverted bottleneck with GELU,
LayerScale and residual connection.  This block forms the encoder of
ConvNeXt-PIV and is the component introduced by the authors.

The inference-side code in this release (`predict.py`, `evaluate.py`,
`pivio.py`, `metrics.py`, `viz.py`, and the command-line/documentation
structure) was written specifically for this repository and is not copied from
RAFT-PIV.

To make the provenance explicit, original and third-party network code are
kept in **separate files**:

| file | origin | license |
| --- | --- | --- |
| `convnext_original.py` | original block of this work | Research Use Only |
| `raft_piv.py` | RAFT-PIV/RAFT components | BSD-3-Clause notice retained |
| `model.py` | assembly only: encoder layout + top-level `PIVNet` | Research Use Only |
| `predict.py`, `evaluate.py`, `pivio.py`, `metrics.py`, `viz.py` | original release code | Research Use Only |

`model.py` reproduces the module and parameter layout required by the trained
checkpoint; the components it imports from `raft_piv.py` follow RAFT-PIV/RAFT
and are attributed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
They are not claimed as original work.

## Third-party code and data

This work builds on the RAFT-PIV architecture and uses the public PIV
benchmark data of Cai et al.  Please read
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full statements and
license texts.

* **RAFT-PIV** — Lagemann, Lagemann, Mukherjee & Schröder,
  *Deep recurrent optical flow learning for particle image velocimetry data*,
  Nature Machine Intelligence 3, 641–651 (2021),
  doi:10.1038/s42256-021-00369-0.  Code Ocean capsule 7226151.
  The correlation pyramid, ConvGRU update block, motion encoder and convex
  upsampling follow this work.
* **RAFT** — Teed & Deng, *RAFT: Recurrent All-Pairs Field Transforms for
  Optical Flow*, ECCV 2020.  BSD-3-Clause.  RAFT-PIV is derived from RAFT.
* **ConvNeXt** — Liu et al., *A ConvNet for the 2020s*, CVPR 2022.  MIT.
  The block design was used as a reference.
* **Data** — the five example pairs are a minimal excerpt from the Cai et al.
  PIV benchmark (link above).  According to the dataset's README, it is
  provided **for research purposes only; all rights reserved; commercial use
  requires the authors' consent**, and they are **not** covered by this
  repository's license.  The vortex-pair and jet data are not part of this
  repository; see [`examples/README.md`](examples/README.md) and the links
  above.

## License

The original code, trained weights, and documentation in this repository are
released under the [Research Use Only License](LICENSE): they may be used,
copied, modified, and distributed for non-commercial research and educational
purposes only.  Commercial use requires prior written permission from the
authors.

Third-party components and data remain under their own terms.  In particular,
`raft_piv.py` contains components derived from RAFT/RAFT-PIV and retains the
BSD 3-Clause notice; the five example image pairs are research-use-only data
from the Cai et al. benchmark and are not covered by this repository's
license.  See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`examples/README.md`](examples/README.md).

## Citation

The citation for this paper will be added upon publication.

Please cite RAFT-PIV when discussing the underlying architecture:

```bibtex
@article{lagemann2021raftpiv,
  title   = {Deep recurrent optical flow learning for particle image
             velocimetry data},
  author  = {Lagemann, Christian and Lagemann, Kai and Mukherjee, Sach and
             Schr{\"o}der, Wolfgang},
  journal = {Nature Machine Intelligence},
  volume  = {3},
  pages   = {641--651},
  year    = {2021},
  doi     = {10.1038/s42256-021-00369-0}
}
```

## Contact

Please open an issue on GitHub for questions about running the code.

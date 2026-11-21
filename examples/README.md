# Sample data

Five sample image pairs are included for a quick check that the released
checkpoint loads and the inference code runs.  They are sample data for code
testing; the benchmark evaluation uses the 10% held-out split described in
`../REPRODUCING.md`.

Each case uses the same simple file naming:

```
${case}_img1.tif   first particle image, 8-bit grayscale, 256 x 256
${case}_img2.tif   second particle image
${case}_flow.flo   ground-truth displacement field (pixels)
```

| folder | files | case |
| --- | --- | --- |
| `five_cases/backstep/` | `backstep_img1.tif`, `backstep_img2.tif`, `backstep_flow.flo` | backward-facing step |
| `five_cases/cylinder/` | `cylinder_img1.tif`, `cylinder_img2.tif`, `cylinder_flow.flo` | flow past a cylinder |
| `five_cases/jhtdb_channel/` | `jhtdb_channel_img1.tif`, `jhtdb_channel_img2.tif`, `jhtdb_channel_flow.flo` | turbulent channel flow |
| `five_cases/dns_turbulence/` | `dns_turbulence_img1.tif`, `dns_turbulence_img2.tif`, `dns_turbulence_flow.flo` | homogeneous isotropic turbulence |
| `five_cases/sqg/` | `sqg_img1.tif`, `sqg_img2.tif`, `sqg_flow.flo` | surface quasi-geostrophic turbulence |

## Quick check

```bash
python evaluate.py --data-dir examples/five_cases --out outputs/sample_check
```

The script writes `metrics.csv` and `summary.csv`.  This is a code check on
the five sample pairs; for the benchmark evaluation, use the complete test
split as described in `../REPRODUCING.md`.

## Data source and terms

The sample pairs are from the public PIV benchmark dataset accompanying

* S. Cai, S. Zhou, C. Xu, Q. Gao, *Dense motion estimation of particle images
  via a convolutional neural network*, Experiments in Fluids **60**, 73
  (2019), doi:[10.1007/s00348-019-2717-2](https://doi.org/10.1007/s00348-019-2717-2).
* Dataset repository: <https://github.com/shengzesnail/PIV_dataset>.

According to the dataset's README, it is **provided for research purposes
only; all rights reserved; commercial use requires the authors' consent**.
The five pairs here are a minimal excerpt.  They are **not** covered by the
repository license.  For the full dataset, use the official repository above
and follow its terms.

The 1997 vortex-pair and jet-flow experimental data used in the paper are not
included in this folder; their public sources are listed in the main
[`README.md`](../README.md#public-datasets-used-in-the-paper).

# Reproducing the benchmark evaluation

This repository contains the model implementation, the trained UCNX-PIV
checkpoint, the evaluation pipeline, the fixed 10% held-out split, and five
sample image pairs for a quick code check.  The commands below run the
benchmark evaluation on that split.

## 1. Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 2.x is recommended.  A CUDA GPU is optional; CPU inference gives the
same values, only slower.

## 2. Quick check with the five sample pairs

```bash
python evaluate.py --data-dir examples/five_cases --out outputs/sample_check
```

The five pairs in `examples/five_cases/` are sample data for checking that the
code runs and the checkpoint loads.  The script writes `metrics.csv` and
`summary.csv`.

## 3. Full benchmark evaluation

The 10% held-out split is specified by `splits/test_seed1.txt`.

Download the public Cai et al. dataset (see the main README for the link) and
arrange it as:

```
PIV_dataset/PIV-genImages/data/
├── backstep/
│   ├── backstep_Re800_00001_img1.tif
│   ├── backstep_Re800_00001_img2.tif
│   └── backstep_Re800_00001_flow.flo
├── cylinder/
├── JHTDB_channel/
├── DNS_turbulence/
└── SQG/
```

Then run:

```bash
python evaluate.py \
  --data-dir /path/to/PIV_dataset/PIV-genImages/data \
  --split-file splits/test_seed1.txt \
  --out outputs/test_split
```

The script checks that every file in the split exists before starting, then
writes:

* `outputs/test_split/metrics.csv` — one row per pair;
* `outputs/test_split/summary.csv` — per-case means and the equal-weight mean.

The split can be regenerated from the dataset with:

```bash
python splits/make_split.py \
  --data-dir /path/to/PIV_dataset/PIV-genImages/data \
  --out splits/test_seed1.txt
```

## 4. Qualitative experimental cases

The vortex-pair and jet-flow images are public but have no ground truth.
Download them from the sources listed in the main README and run:

```bash
# Vortex pair (Willert & Gharib, 1997)
python predict.py \
  --img1 vortexPair_img1.tif \
  --img2 vortexPair_img2.tif \
  --out outputs/vortex_pair

# PIV Challenge jet flow
python predict.py \
  --img1 c050a.bmp \
  --img2 c050b.bmp \
  --out outputs/jet
```

`predict.py` writes the predicted `.flo`, a colour-coded PNG, and a
side-by-side panel.  The experimental images are not redistributed in this
repository.

## 5. Checksums

* `weights/SHA256SUMS.txt` — released checkpoint.
* `examples/SHA256SUMS.txt` — the 15 sample files (five pairs).

Verify with:

```bash
cd examples && sha256sum -c SHA256SUMS.txt
cd ../weights && sha256sum -c SHA256SUMS.txt
```

## Scope

The released checkpoint, implementation, and evaluation pipeline compute the
per-case metrics and the equal-weight mean on the 10% held-out split specified
by `splits/test_seed1.txt`.  The public benchmark images are third-party
materials and are obtained from their original sources as described in the
main README.

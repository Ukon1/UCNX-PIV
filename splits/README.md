# Test split

`test_seed1.txt` is the fixed 10% held-out split used by the evaluation code.
It contains one relative `*_img1.tif` path per line.

The split was generated **per case** from the alphabetically sorted
`*_img1.tif` list with:

```python
train_size = int(0.9 * N)
val_size = int(0.1 * N)
train, val = torch.utils.data.random_split(
    files, [train_size, val_size],
    generator=torch.Generator().manual_seed(1),
)
```

The paths in the file are relative to the dataset root
(`PIV_dataset/PIV-genImages/data`).  They are sorted for readability; the
membership is what matters.

To regenerate or verify the file from a local copy of the dataset:

```bash
python splits/make_split.py \
  --data-dir /path/to/PIV_dataset/PIV-genImages/data \
  --out splits/test_seed1.txt
```

`make_split.py` writes the same paths.

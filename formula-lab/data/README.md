# `data/` — datasets

Datasets are **not** committed to the repository (see `.gitignore`). Put data
here locally.

## To use the real data

This project targets the Kaggle dataset **"Hong Kong Horse Racing"** by Graham
Daley: https://www.kaggle.com/datasets/gdaley/hkracing

Place these two files in `data/raw/`:

```
data/raw/races.csv
data/raw/runs.csv
```

You can get them either way:

- **Automatically:** `python scripts/download_kaggle.py`
  (requires `pip install kaggle` and a `kaggle.json` API token — the script
  prints exact setup steps).
- **Manually:** download the two CSVs from the dataset page and drop them in.

The loader (`src/data/loader.py`) automatically prefers real CSVs in
`data/raw/` over synthetic data. No code change is needed.

## If you have no data

That's fine — `src/data/synthetic.py` generates a dataset with the **same
schema**, and everything downstream runs unchanged. The synthetic numbers are
illustrative of the machinery only; see
[`docs/ASSUMPTIONS_LIMITATIONS.md`](../docs/ASSUMPTIONS_LIMITATIONS.md).

## Folders

- `data/raw/` — original CSVs (real or synthetic cache). Git-ignored.
- `data/processed/` — any derived/cached artifacts you choose to write.
  Git-ignored.

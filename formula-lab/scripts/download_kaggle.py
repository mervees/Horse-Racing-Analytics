#!/usr/bin/env python
"""Download the Hong Kong horse-racing dataset from Kaggle into data/raw/.

This project is built against the schema of the public Kaggle dataset
"Hong Kong Horse Racing" by Graham Daley:

    https://www.kaggle.com/datasets/gdaley/hkracing

It provides two CSVs that join on `race_id`:
  - races.csv : one row per race  (venue, date, distance, going, ...)
  - runs.csv  : one row per runner (horse_id, jockey_id, draw, win_odds, result, ...)

------------------------------------------------------------------------
SETUP (one time)
------------------------------------------------------------------------
1. Create a free Kaggle account.
2. Go to https://www.kaggle.com/settings  ->  "API"  ->  "Create New Token".
   This downloads a `kaggle.json` file.
3. Place it where the Kaggle CLI expects it:
       Linux/macOS:  ~/.kaggle/kaggle.json   (chmod 600)
       Windows:      %USERPROFILE%\\.kaggle\\kaggle.json
   (Alternatively set KAGGLE_USERNAME and KAGGLE_KEY environment variables.)
4. Install the client:  pip install kaggle

Then run:
    python scripts/download_kaggle.py

If you prefer to do it manually, just download the two CSVs from the dataset
page and drop them into data/raw/ — the loader will pick them up with no code
change. If neither real CSVs nor the Kaggle CLI are available, the rest of the
project still runs on synthetic data that matches this exact schema.
"""
from __future__ import annotations

import os
import sys

DATASET = "gdaley/hkracing"
DEST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


def main() -> int:
    os.makedirs(DEST, exist_ok=True)

    try:
        # Importing here triggers Kaggle's credential check.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print(
            "The `kaggle` package is not installed.\n"
            "  pip install kaggle\n"
            "Then follow the setup notes at the top of this file, or download\n"
            f"the CSVs manually from https://www.kaggle.com/datasets/{DATASET}\n"
            f"and place races.csv + runs.csv into {DEST}",
            file=sys.stderr,
        )
        return 1

    try:
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {DATASET} -> {DEST}")
        api.dataset_download_files(DATASET, path=DEST, unzip=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Kaggle download failed: {exc}\n"
            "Check your kaggle.json credentials (Account -> API -> Create New Token),\n"
            "or download the CSVs manually from the dataset page.",
            file=sys.stderr,
        )
        return 1

    expected = ["races.csv", "runs.csv"]
    missing = [f for f in expected if not os.path.exists(os.path.join(DEST, f))]
    if missing:
        print(f"Warning: expected files not found after download: {missing}", file=sys.stderr)
        return 1

    print("Done. Found races.csv and runs.csv in", DEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

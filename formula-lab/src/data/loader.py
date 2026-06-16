"""
Load, clean, and join the racing data into a single runner-grain table.

Priority:
1. Real Kaggle files in ``data/raw/`` (``races.csv`` + ``runs.csv``).
2. Otherwise, generate a synthetic dataset (and optionally cache it to raw).

The public entry point is :func:`load_dataset`, which returns a tidy
``(runs, races)`` pair plus a merged ``runner-grain`` frame ready for feature
engineering. Cleaning is intentionally conservative -- we coerce types, parse
dates, and drop structurally broken rows, but we do not impute aggressively
here; leakage-safe imputation happens inside the feature layer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema
from .synthetic import write_synthetic


@dataclass
class Dataset:
    """Container for the loaded, cleaned data."""

    runs: pd.DataFrame      # runner-grain (runs.csv), cleaned
    races: pd.DataFrame     # race-grain (races.csv), cleaned
    merged: pd.DataFrame    # runs joined with race-level columns, sorted by date
    source: str             # "kaggle" or "synthetic"

    @property
    def n_races(self) -> int:
        return self.merged[schema.RACE_KEY].nunique()

    @property
    def n_runs(self) -> int:
        return len(self.merged)


def _apply_map(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clean_races(races: pd.DataFrame) -> pd.DataFrame:
    races = _apply_map(races, schema.DEFAULT_COLUMN_MAP.races)
    if "date" in races.columns:
        races["date"] = pd.to_datetime(races["date"], errors="coerce")
    races = _coerce_numeric(
        races, ["race_no", "surface", "distance", "race_class", "prize"]
    )
    races = races.dropna(subset=[schema.RACE_KEY]).copy()
    races[schema.RACE_KEY] = races[schema.RACE_KEY].astype(int)
    return races.drop_duplicates(subset=[schema.RACE_KEY])


def _clean_runs(runs: pd.DataFrame) -> pd.DataFrame:
    runs = _apply_map(runs, schema.DEFAULT_COLUMN_MAP.runs)
    runs = _coerce_numeric(
        runs,
        [
            "horse_no", "horse_id", "result", "won", "horse_age", "horse_rating",
            "declared_weight", "actual_weight", "draw", "trainer_id", "jockey_id",
            "win_odds", "place_odds", "finish_time",
        ],
    )
    # Structural requirements: must identify the race, the horse, and the result.
    runs = runs.dropna(subset=[schema.RACE_KEY, "horse_id"]).copy()
    runs[schema.RACE_KEY] = runs[schema.RACE_KEY].astype(int)
    runs["horse_id"] = runs["horse_id"].astype(int)

    # Some providers code "did not finish" as 0 / 99; treat non-positive results
    # as missing rank but keep the row (the horse still ran).
    if "result" in runs.columns:
        runs.loc[runs["result"] <= 0, "result"] = np.nan
    if "won" not in runs.columns and "result" in runs.columns:
        runs["won"] = (runs["result"] == 1).astype("Int64")

    # Odds must be > 1.0 to be valid decimal odds; clip absurd values.
    for c in ("win_odds", "place_odds"):
        if c in runs.columns:
            runs.loc[runs[c] <= 1.0, c] = np.nan
            runs.loc[runs[c] > 999, c] = np.nan
    return runs


def load_dataset(
    raw_dir: str = "data/raw",
    *,
    allow_synthetic: bool = True,
    cache_synthetic: bool = True,
    synthetic_kwargs: dict | None = None,
) -> Dataset:
    """Load the dataset, preferring real Kaggle files in ``raw_dir``.

    Returns a :class:`Dataset`. Set ``allow_synthetic=False`` to hard-require the
    real files (raises ``FileNotFoundError`` if missing).
    """
    races_path = os.path.join(raw_dir, "races.csv")
    runs_path = os.path.join(raw_dir, "runs.csv")
    have_real = os.path.exists(races_path) and os.path.exists(runs_path)

    if have_real:
        source = "kaggle"
    elif allow_synthetic:
        if cache_synthetic:
            write_synthetic(raw_dir, **(synthetic_kwargs or {}))
        else:
            from .synthetic import generate

            races, runs = generate(**(synthetic_kwargs or {}))
            return _finalize(_clean_runs(runs), _clean_races(races), "synthetic")
        source = "synthetic"
    else:
        raise FileNotFoundError(
            f"Expected races.csv and runs.csv in {raw_dir!r}. "
            "Run `python scripts/download_kaggle.py` or pass allow_synthetic=True."
        )

    races = _clean_races(pd.read_csv(races_path))
    runs = _clean_runs(pd.read_csv(runs_path))
    return _finalize(runs, races, source)


def _finalize(runs: pd.DataFrame, races: pd.DataFrame, source: str) -> Dataset:
    race_cols = [c for c in schema.RACE_PRE_RACE_COLS if c in races.columns]
    merged = runs.merge(races[race_cols], on=schema.RACE_KEY, how="left")

    # Canonical sort: chronological, then by race, then by saddle number.
    sort_cols = [c for c in ["date", schema.RACE_KEY, "horse_no"] if c in merged]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # field_size is a genuinely pre-race quantity (declared runners) and is heavily
    # used downstream, so compute it once here.
    merged["field_size"] = merged.groupby(schema.RACE_KEY)["horse_id"].transform("count")
    return Dataset(runs=runs, races=races, merged=merged, source=source)


if __name__ == "__main__":
    ds = load_dataset()
    print(f"source={ds.source}  races={ds.n_races}  runs={ds.n_runs}")
    print(ds.merged.head())

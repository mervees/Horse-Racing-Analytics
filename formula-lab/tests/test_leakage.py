"""Tests for the no-leakage guarantees.

Leakage (letting post-race or same-race information into the features) is the
single most common way a horse-racing model looks brilliant in backtests and
loses money live. These tests encode the guarantees the feature builder and
the backtest split must uphold.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.schema import LEAKAGE_COLS, LABEL_WON, LABEL_RESULT
from src.features.build_features import build_features
from src.backtest.backtest import time_split


@pytest.fixture(scope="module")
def bundle():
    ds = load_dataset(allow_synthetic=True, synthetic_kwargs=dict(n_races=300, seed=3))
    return build_features(ds.merged, include_market=False)


def test_no_result_columns_in_features(bundle):
    """Result/settlement columns must never appear in the feature matrix."""
    banned = set(LEAKAGE_COLS) | {LABEL_WON, LABEL_RESULT}
    leaked = [c for c in bundle.feature_cols if c in banned]
    assert leaked == [], f"leakage columns leaked into features: {leaked}"


def test_market_excluded_by_default(bundle):
    """With include_market=False, raw odds must not be features."""
    for col in ("win_odds", "place_odds"):
        assert col not in bundle.feature_cols


def test_no_feature_trivially_determines_label(bundle):
    """No single numeric feature should correlate near-perfectly with winning."""
    num = bundle.X.select_dtypes("number").copy()
    num["_y"] = bundle.y.values
    corr = num.corr()["_y"].drop("_y").abs()
    worst = corr.max()
    assert worst < 0.6, f"suspiciously high feature/label correlation: {corr.idxmax()}={worst:.3f}"


def test_time_split_is_chronological(bundle):
    """Every training row must occur no later than every test row."""
    train_idx, test_idx = time_split(bundle, train_frac=0.7)
    assert len(train_idx) > 0 and len(test_idx) > 0
    train_max = bundle.date.iloc[train_idx].max()
    test_min = bundle.date.iloc[test_idx].min()
    assert train_max <= test_min, "training data bleeds past the start of the test window"


def test_no_race_spans_both_splits(bundle):
    """A single race must live entirely in train or entirely in test."""
    train_idx, test_idx = time_split(bundle, train_frac=0.7)
    train_races = set(bundle.groups.iloc[train_idx])
    test_races = set(bundle.groups.iloc[test_idx])
    assert train_races.isdisjoint(test_races)


def test_debut_runners_have_zero_prior_history(bundle):
    """A horse's first-ever run must show zero career runs (no peeking ahead)."""
    frame = bundle.frame
    if "horse_career_runs" in frame and "is_debut" in frame:
        debuts = frame[frame["is_debut"] == 1]
        assert (debuts["horse_career_runs"] == 0).all()

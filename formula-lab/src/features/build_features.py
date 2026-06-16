"""
Feature engineering -- leakage-safe by construction.

The cardinal rule of racing models: a feature for race *R* may only use
information available *before R goes off*. Every rolling/career statistic here is
computed over rows strictly earlier in chronological order, using the
"cumulative-minus-current" and "shift-then-roll" patterns, so the current run is
never folded into its own features.

Two feature groups are produced:

* **fundamentals** -- horse / jockey / trainer form, draw, weight, distance, going,
  class. These describe the horse's chance independent of the betting market.
* **market**        -- features derived from pre-off odds. Kept separate so you can
  train a model that is *independent of the market* and then look for value by
  comparing model probability against market probability. Set
  ``include_market=True`` to fold them into the model instead.

``build_features`` returns an :class:`FeatureBundle` with the matrix, the label,
the per-race group key (for ranking + grouped CV), and metadata listing which
columns are categorical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import schema

CATEGORICAL = ["venue", "going", "config", "horse_type", "horse_country", "surface"]


@dataclass
class FeatureBundle:
    X: pd.DataFrame
    y: pd.Series                 # 1 = won
    result: pd.Series            # finishing position (for ranking labels / NDCG)
    groups: pd.Series            # race_id, defines a "query group" for ranking
    date: pd.Series
    feature_cols: list[str]
    categorical_cols: list[str]
    market_cols: list[str]
    frame: pd.DataFrame          # everything joined back together, for reporting


def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).fillna(0.0)


def _prior_career(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Career counts/rates for ``key`` using only prior runs (no leakage)."""
    g = df.groupby(key, sort=False)
    runs = g.cumcount()                                   # # of strictly-prior runs
    won = df["won"].fillna(0).astype(float)
    top3 = (df["result"] <= 3).fillna(False).astype(float)
    prior_wins = g["won"].cumsum().fillna(0) - won
    prior_top3 = top3.groupby(df[key]).cumsum() - top3
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_runs"] = runs
    out[f"{prefix}_win_rate"] = _safe_rate(prior_wins, runs)
    out[f"{prefix}_place_rate"] = _safe_rate(prior_top3, runs)
    return out


def _prior_form(df: pd.DataFrame, key: str, prefix: str, window: int) -> pd.DataFrame:
    """Rolling recent form (mean win, mean finishing position) over the last
    ``window`` prior runs."""
    g = df.groupby(key, sort=False)
    won_form = g["won"].transform(
        lambda s: s.shift().rolling(window, min_periods=1).mean()
    )
    pos_form = g["result"].transform(
        lambda s: s.shift().rolling(window, min_periods=1).mean()
    )
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_form_win_{window}"] = won_form
    out[f"{prefix}_form_pos_{window}"] = pos_form
    return out


def build_features(
    merged: pd.DataFrame,
    *,
    include_market: bool = False,
    drop_no_history: bool = False,
) -> FeatureBundle:
    """Build the modelling matrix from the merged runner-grain frame.

    ``include_market`` folds odds-derived features into the model. Default False:
    the model stays market-independent so you can hunt for value.
    ``drop_no_history`` removes first-ever runs (all-zero history) if you prefer
    not to score debutants.
    """
    df = merged.sort_values(["date", schema.RACE_KEY, "horse_no"]).reset_index(drop=True)

    parts: list[pd.DataFrame] = []

    # --- entity form (horse / jockey / trainer) -----------------------------
    parts.append(_prior_career(df, "horse_id", "horse"))
    parts.append(_prior_form(df, "horse_id", "horse", 3))
    if "jockey_id" in df.columns:
        parts.append(_prior_career(df, "jockey_id", "jockey"))
        parts.append(_prior_form(df, "jockey_id", "jockey", 20))
    if "trainer_id" in df.columns:
        parts.append(_prior_career(df, "trainer_id", "trainer"))
        parts.append(_prior_form(df, "trainer_id", "trainer", 20))

    feat = pd.concat([df] + parts, axis=1)

    # --- recency / freshness -------------------------------------------------
    feat["horse_days_since"] = (
        feat.groupby("horse_id")["date"].diff().dt.days
    )
    # First-start indicator (informative on its own).
    feat["is_debut"] = (feat["horse_runs"] == 0).astype(int)

    # --- draw / weight / distance -------------------------------------------
    if "draw" in feat.columns:
        feat["draw_norm"] = feat["draw"] / feat["field_size"].clip(lower=1)
    if "actual_weight" in feat.columns:
        race_mean_w = feat.groupby(schema.RACE_KEY)["actual_weight"].transform("mean")
        feat["weight_vs_field"] = feat["actual_weight"] - race_mean_w
    if "distance" in feat.columns:
        feat["is_sprint"] = (feat["distance"] <= 1200).astype(int)
        feat["is_route"] = (feat["distance"] >= 1800).astype(int)
    if "horse_rating" in feat.columns:
        race_mean_r = feat.groupby(schema.RACE_KEY)["horse_rating"].transform("mean")
        feat["rating_vs_field"] = feat["horse_rating"] - race_mean_r

    # --- market features (optional) -----------------------------------------
    market_cols: list[str] = []
    if "win_odds" in feat.columns:
        feat["log_win_odds"] = np.log(feat["win_odds"].clip(lower=1.01))
        # Raw implied prob (un-normalised); the proper de-vigged version lives in
        # the probability module. This is just a model feature.
        feat["mkt_implied_raw"] = 1.0 / feat["win_odds"].clip(lower=1.01)
        # Favouritism rank within the race (1 = shortest price).
        feat["mkt_fav_rank"] = feat.groupby(schema.RACE_KEY)["win_odds"].rank(
            method="min"
        )
        market_cols = ["log_win_odds", "mkt_implied_raw", "mkt_fav_rank"]

    # --- assemble feature list ----------------------------------------------
    fundamental_cols = [
        "horse_runs", "horse_win_rate", "horse_place_rate",
        "horse_form_win_3", "horse_form_pos_3", "horse_days_since", "is_debut",
        "jockey_runs", "jockey_win_rate", "jockey_form_win_20",
        "trainer_runs", "trainer_win_rate", "trainer_form_win_20",
        "draw", "draw_norm", "field_size",
        "actual_weight", "weight_vs_field",
        "distance", "is_sprint", "is_route",
        "horse_age", "horse_rating", "rating_vs_field",
    ]
    fundamental_cols = [c for c in fundamental_cols if c in feat.columns]
    cat_cols = [c for c in CATEGORICAL if c in feat.columns]
    for c in cat_cols:
        feat[c] = feat[c].astype("category")

    feature_cols = fundamental_cols + cat_cols
    if include_market:
        feature_cols = feature_cols + market_cols

    if drop_no_history:
        feat = feat[feat["horse_runs"] > 0].reset_index(drop=True)

    # Numeric features: fill remaining NaNs with a neutral 0 (rates already 0;
    # days_since NaN -> debut, encode as a large value so "long layoff" != "fresh").
    if "horse_days_since" in feat.columns:
        feat["horse_days_since"] = feat["horse_days_since"].fillna(9999)
    num_feats = [c for c in feature_cols if c not in cat_cols]
    feat[num_feats] = feat[num_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y = feat["won"].fillna(0).astype(int)
    result = feat["result"]
    return FeatureBundle(
        X=feat[feature_cols],
        y=y,
        result=result,
        groups=feat[schema.RACE_KEY],
        date=feat["date"],
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        market_cols=market_cols,
        frame=feat,
    )

"""
Canonical schema for the Hong Kong horse-racing dataset (Graham Daley `hkracing`
on Kaggle: https://www.kaggle.com/datasets/gdaley/hkracing).

The dataset ships as two CSVs that join on ``race_id``:

* ``races.csv`` -- one row per race (track, distance, going, dividends, ...).
* ``runs.csv``  -- one row per horse-run (one runner in one race).

This module is the single source of truth for column names so that the rest of
the pipeline never hard-codes raw strings. If you swap in a different provider's
data, adapt the ``RAW_*`` lists and the ``rename_*`` maps here and the downstream
code keeps working.

We deliberately separate three classes of columns:

1.  PRE-RACE columns  -- known *before* the gates open. Only these may feed a model.
2.  RESULT columns    -- the outcome (finishing position, win flag, finish time).
3.  SETTLEMENT columns -- dividends/odds that settle after betting closes.

Mixing (2)/(3) into features is the #1 way to build a model that looks brilliant
in backtests and loses money live. ``features.build_features`` enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# --------------------------------------------------------------------------- #
# races.csv
# --------------------------------------------------------------------------- #
RACE_KEY: Final = "race_id"

# Columns that describe the race itself and are known in advance.
RACE_PRE_RACE_COLS: Final = [
    "race_id",
    "date",
    "venue",        # ST = Sha Tin, HV = Happy Valley
    "race_no",
    "config",       # track configuration, e.g. "A", "A+3", "B", "C", "C+3"
    "surface",      # 0 = turf, 1 = all-weather (provider-specific)
    "distance",     # metres: 1000/1200/1400/1600/1650/1800/2000/2200/2400
    "going",        # "GOOD", "GOOD TO FIRM", "YIELDING", "WET SLOW", ...
    "horse_ratings",  # rating band of the race, e.g. "40-60"
    "race_class",     # class 1 (best) .. class 5, plus group/listed
    "prize",          # total prize money
]

# Section + sectional finishing times for the race (post-result).
RACE_RESULT_COLS: Final = [f"sec_time{i}" for i in range(1, 8)] + [
    f"time{i}" for i in range(1, 8)
]

# Tote dividends (settle after betting closes).
RACE_SETTLEMENT_COLS: Final = (
    [f"place_combination{i}" for i in range(1, 5)]
    + [f"place_dividend{i}" for i in range(1, 5)]
    + [f"win_combination{i}" for i in range(1, 3)]
    + [f"win_dividend{i}" for i in range(1, 3)]
)

# --------------------------------------------------------------------------- #
# runs.csv
# --------------------------------------------------------------------------- #
RUN_PRE_RACE_COLS: Final = [
    "race_id",
    "horse_no",        # saddle-cloth number in this race
    "horse_id",
    "horse_age",
    "horse_country",
    "horse_type",      # Gelding / Mare / Horse / Colt / Filly / Rig
    "horse_rating",    # official rating of the horse going into the race
    "horse_gear",      # blinkers, visor, etc. (string codes)
    "declared_weight",  # horse + jockey, lbs
    "actual_weight",    # weight carried (handicap), lbs
    "draw",             # post / barrier position
    "trainer_id",
    "jockey_id",
]

# Pre-race market signal. Odds are *known before the off* and are the single most
# informative pre-race feature in racing -- but they are also what we try to beat,
# so we treat them as a feature AND as a benchmark, never as the label.
RUN_MARKET_COLS: Final = ["win_odds", "place_odds"]

RUN_RESULT_COLS: Final = (
    ["result", "won", "finish_time"]
    + [f"position_sec{i}" for i in range(1, 7)]
    + [f"behind_sec{i}" for i in range(1, 7)]
    + [f"time{i}" for i in range(1, 7)]
)

# --------------------------------------------------------------------------- #
# Derived groupings used throughout the pipeline
# --------------------------------------------------------------------------- #
#: Everything a model is allowed to see at prediction time.
LEGAL_PRE_RACE_COLS: Final = sorted(
    set(RACE_PRE_RACE_COLS + RUN_PRE_RACE_COLS + RUN_MARKET_COLS)
)

#: Anything in here must never become a feature (post-outcome leakage).
LEAKAGE_COLS: Final = sorted(
    set(RACE_RESULT_COLS + RACE_SETTLEMENT_COLS + RUN_RESULT_COLS) - {"race_id"}
)

#: The supervised label.
LABEL_WON: Final = "won"
LABEL_RESULT: Final = "result"


@dataclass(frozen=True)
class ColumnMap:
    """Maps an arbitrary provider's column names onto the canonical names.

    Override the fields when ingesting a non-HKJC dataset. Anything not listed is
    passed through unchanged.
    """

    races: dict[str, str] = field(default_factory=dict)
    runs: dict[str, str] = field(default_factory=dict)


# Default map: identity for the gdaley/hkracing files. A handful of common
# alternate spellings are included so other Kaggle mirrors load without edits.
DEFAULT_COLUMN_MAP: Final = ColumnMap(
    races={
        "raceId": "race_id",
        "raceNo": "race_no",
        "raceClass": "race_class",
    },
    runs={
        "raceId": "race_id",
        "horseId": "horse_id",
        "horseNo": "horse_no",
        "jockeyId": "jockey_id",
        "trainerId": "trainer_id",
        "winOdds": "win_odds",
        "placeOdds": "place_odds",
        "actualWeight": "actual_weight",
        "declaredWeight": "declared_weight",
        "finishTime": "finish_time",
    },
)

# Data Structure

The lab is built against the schema of the public Kaggle dataset **"Hong Kong
Horse Racing"** by Graham Daley
([gdaley/hkracing](https://www.kaggle.com/datasets/gdaley/hkracing)). It is two
CSV tables that join on `race_id`.

If you do not have the real data, `src/data/synthetic.py` generates a dataset
with the **same column names and dtypes**, so every downstream module runs
unchanged. The loader prefers real CSVs in `data/raw/` and silently falls back
to synthetic data otherwise.

## `races.csv` — one row per race

| column        | meaning                                            |
|---------------|----------------------------------------------------|
| `race_id`     | primary key, joins to `runs.csv`                   |
| `date`        | race date                                          |
| `venue`       | course (e.g. Sha Tin, Happy Valley)                |
| `race_no`     | race number on the card                            |
| `config`      | track configuration                                |
| `surface`     | turf / all-weather                                 |
| `distance`    | race distance in metres                            |
| `going`       | track condition (e.g. GOOD, YIELDING)              |
| `race_class`  | class/grade of the race                            |
| `prize`       | total prize money                                  |

## `runs.csv` — one row per runner (horse-in-race)

The crucial distinction for honest modelling is **when** a column becomes known:

### Pre-race (legal model inputs — known before the off)
| column            | meaning                                        |
|-------------------|------------------------------------------------|
| `race_id`         | foreign key to `races.csv`                     |
| `horse_id`        | horse identifier                               |
| `jockey_id`       | jockey identifier                              |
| `trainer_id`      | trainer identifier                             |
| `draw`            | barrier/gate number                            |
| `declared_weight` | declared weight (horse + gear)                 |
| `actual_weight`   | weight carried (handicap)                      |
| `horse_age`       | age in years                                   |
| `horse_rating`    | official rating going into the race            |
| `horse_country`   | country of origin                              |
| `horse_type`      | gelding / colt / mare etc.                     |
| `horse_gear`      | gear worn (blinkers, etc.)                     |

### Market (known pre-race, but treated separately — see note)
| column        | meaning                          |
|---------------|----------------------------------|
| `win_odds`    | final win odds (decimal)         |
| `place_odds`  | final place odds (decimal)       |

### Result / settlement (LEAKAGE — known only after the race)
| column                | meaning                                       |
|-----------------------|-----------------------------------------------|
| `result`              | finishing position (1 = winner)               |
| `won`                 | 1 if the horse won, else 0                     |
| `position_sec1..6`    | in-running sectional positions                |
| `behind_sec1..6`      | lengths behind at each sectional              |
| `finish_time`         | finishing time                                |
| `lengths_behind`      | margin behind the winner                      |

`src/data/schema.py` is the single source of truth for these groupings. It
exposes `LEGAL_PRE_RACE_COLS`, `LEAKAGE_COLS`, `LABEL_WON`, `LABEL_RESULT`, and
`RACE_KEY`, plus a `ColumnMap` for providers that spell columns differently.

## Why market odds are kept separate

Final odds are known before the off, so they are technically "legal" features.
But folding them into the model makes the model *imitate the market* — which
is already very efficient. The whole point of value betting is to find spots
where **our independent estimate disagrees with the market**. So by default
(`include_market=False`) the model never sees odds; odds enter only afterwards,
in the probability layer, where we compare model probability against the
de-vigged market probability. You can set `include_market=True` to study how
much the market price improves pure predictive accuracy (it improves it a lot —
that is exactly why it is dangerous as a feature for value-hunting).

## The runner-grain "merged" frame

`load_dataset()` returns a `Dataset` with `runs`, `races`, and a `merged`
frame. `merged` is `runs` left-joined to `races` on `race_id`, **sorted
chronologically**, with a derived `field_size` (runners per race). This
chronological ordering is what makes the "prior-only" rolling features and the
time-based backtest split correct.

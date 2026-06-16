# Model Design

There are two complementary scoring systems in the lab, plus a leakage-safe
feature layer that feeds them.

## 1. Feature engineering (`src/features/build_features.py`)

This is the correctness-critical module. Every feature is built from
information that would have been available **before** the race, using two
patterns:

- **cumulative-minus-current**: a horse's career win rate at race *t* is
  computed from races strictly before *t*. We take the cumulative count up to
  and including the row, then subtract the current row, so a run never
  contributes to its own features.
- **shift-then-roll**: recent-form windows (last 3, last 20) are computed by
  shifting one row back per entity and then rolling, so the current result is
  excluded from its own rolling average.

Feature families produced:

- **Horse history**: career runs, career win/place rate (prior-only), recent
  win rate and recent average finishing position, days since last run, debut
  flag.
- **Jockey / trainer**: career runs and win/place rates (prior-only). These
  are strong, stable signals in real data.
- **Draw / field**: draw, normalised draw, field size.
- **Weight**: actual weight, weight relative to the field that day.
- **Distance**: distance, sprint/route flags.
- **Class / rating**: official rating, rating relative to the field.
- **Categoricals** (kept as pandas `category` dtype for LightGBM): venue,
  going, config, horse type, horse country, surface.

The output is a `FeatureBundle` carrying the design matrix `X`, the win label
`y`, the finishing `result`, the `groups` (race_id, for grouped ranking and
grouped splitting), the `date` (for chronological splitting), and the column
metadata.

### A note on weather
The HK dataset does not ship a separate weather feed; track **going** (GOOD,
YIELDING, SOFT, …) is the racing-relevant encoding of weather and is included.
If you attach an external weather source, add it to the pre-race column group
in `schema.py` and it will flow through as another feature — the architecture
is designed for exactly this kind of extension.

## 2. Glass-box rule scorer (`src/models/scoring.py`)

A transparent, tunable baseline that needs no training. For each race it:

1. z-scores each signal **within the race** (so horses are compared only to
   today's rivals),
2. inverts "lower-is-better" signals (recent finishing position, weight, draw,
   days-since),
3. takes a weighted sum (`DEFAULT_WEIGHTS`, fully tunable) into a `rule_raw`
   score,
4. converts to a 0–100 `rule_score` (within-race percentile) and a `rule_prob`
   (softmax over the field, so probabilities sum to 1).

`top_signals_for_runner()` returns the few signals that moved a horse's score
most — this is what the explanation layer narrates. The rule scorer is valuable
as an interpretable benchmark and as a sanity check on the ML model.

## 3. Learned ranker (`src/models/ranker.py`)

The main model. Two objectives behind one `RaceRanker` interface:

- **`ranking`** (default): a LightGBM **LambdaRank** model. Relevance is
  `clip(field_size - result, 0, 31)` so the winner gets the highest relevance,
  and learning is grouped by race — the model learns to *order runners within a
  race*, which is exactly the prediction task.
- **`classification`**: a LightGBM binary classifier on `won`.

Both convert raw outputs to **within-race win probabilities that sum to 1**
(softmax for the ranker; renormalised `predict_proba` for the classifier — the
conditional-logit trick). If LightGBM is not installed, it falls back to
sklearn's `HistGradientBoostingClassifier`, encoding categoricals via category
codes. The class exposes `fit`, `predict_proba`, `feature_importance`, and
`save`/`load` (joblib).

## Why two systems

The rule scorer is auditable and never overfits; the learned ranker is more
accurate but a black box. Running both lets you (a) explain predictions in
plain language via the rule signals, (b) detect when the ML model has drifted
far from a sensible baseline, and (c) ensemble or blend if desired.

## Improvement ideas (not yet implemented)

- Calibrated probabilities via isotonic / Platt scaling on a held-out slice.
- Pace/sectional-derived *pre-race* features built from a horse's **past**
  sectionals (never the current race's).
- Trainer–jockey combination effects and course-and-distance specialisation.
- Monotonic constraints in LightGBM (e.g. higher rating should not lower the
  score) for robustness.
- Proper hyperparameter search with a purged, embargoed time-series CV.

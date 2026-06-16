# Probability Logic

Everything in `src/probability/probability.py` turns model scores and market
odds into decision-ready numbers: fair probabilities, edge, stake sizing,
confidence, and risk flags. All of it is deterministic and auditable.

## From odds to probability

- **Implied probability**: `implied_prob(odds) = 1 / decimal_odds`.
- **Book sum / overround**: the implied probabilities across a race sum to more
  than 1; the excess is the bookmaker's margin (the "vig" or "overround").
  `overround_pct()` reports it. In HK win pools this is typically ~17–22%.

## De-vigging (removing the margin)

`fair_probabilities(odds, method=...)` rescales implied probabilities back to
sum to 1. Three methods:

- **`multiplicative`** (default): divide every implied probability by the book
  sum. Simple and robust; assumes the margin is spread proportionally.
- **`power`**: find the exponent *k* such that `sum(p_i^k) = 1` (solved with
  Brent's method). Allows the margin to fall unevenly across the book.
- **`shin`**: the Shin (1992) model, which assumes a fraction *z* of money
  comes from insiders and solves for the fair probabilities accordingly. It
  removes proportionally more margin from longshots, correcting the
  **favourite–longshot bias** — versus naive multiplicative de-vig it nudges
  the favourite's fair probability up and longshots' down. Falls back to
  multiplicative if the solver fails.

These let you choose how skeptically to read the market price before comparing
it to the model.

## Edge and expected value

- **Expected value** per unit staked: `EV = p_model * decimal_odds - 1`. This
  is also the **edge**. Positive EV means the model thinks the price is too big
  (value); negative EV means too short.

## Stake sizing — fractional Kelly

`kelly_fraction(p, odds, fraction, cap)` implements
`f* = (b*p - q) / b` where `b = odds - 1`, `q = 1 - p`. Key safety choices:

- Negative-edge bets return **0** (never stake against yourself).
- `fraction` applies **fractional Kelly** (the pipeline uses 0.25 — quarter
  Kelly — the usual real-world choice, because full Kelly is brutally volatile
  and very sensitive to probability error).
- `cap` hard-limits any single stake (default 10% of bankroll).

## Confidence score (a human aid, not a second model)

`confidence_score()` returns a transparent 0–100 number that rewards: high
absolute model probability (scaled by field size), broad agreement with the
market (wild disagreement is usually an overfitting tell), and sample
sufficiency (a horse with more career runs is on firmer ground). It is a
**triage/ranking aid for a human reviewer**, deliberately simple and explainable
— not an additional predictive layer.

## Risk signals

`risk_signals()` returns per-runner flags to triage bets before placing them:

- `thin_edge` — positive but wafer-thin EV (likely noise),
- `longshot` — odds ≥ 15 (high variance, sensitive to price error),
- `low_history` — fewer than 4 career runs (the model is extrapolating),
- `model_market_divergence` — model prob more than 2.5× the fair market prob
  (overfitting or stale odds),
- `large_field` — 14+ runners (harder to predict),
- `edge` — the raw EV, for sorting.

## One-call pricing

`price_race(runners, model_prob_col, odds_col, devig_method, kelly_fraction_mult)`
annotates a single race's runners with fair market probability, edge, Kelly
stake, confidence, and all risk flags in one pass. The pipeline calls this for
every race in the test set to build the betting view the backtester consumes.

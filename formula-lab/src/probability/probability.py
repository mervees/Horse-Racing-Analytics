"""
Probability, value, and staking logic.

Everything a value-betting system needs to turn (model probability, market odds)
into a decision:

* **implied probability** from decimal odds, and **de-vigging** to recover the
  market's *fair* probabilities (the raw 1/odds over-sum to >1 because of the
  bookmaker margin / tote takeout).
* **fair odds** from any probability.
* **expected value / edge** and per-bet **ROI**.
* **Kelly** stake sizing (with fractional-Kelly safety).
* a transparent **confidence score** and a set of **risk signals**.

De-vigging methods
------------------
* ``multiplicative`` -- divide each implied prob by the booksum. Simple, unbiased
  if the margin is spread proportionally. Default.
* ``power``          -- find exponent k with sum(p_i**k)=1. Corrects the
  favourite-longshot bias (favourites are over-bet, longshots under-bet).
* ``shin``           -- Shin (1992) model with an insider-trading parameter z,
  solved so the fair probs sum to 1. Industry standard for sharp books.

References: Štrumbelj (2014), "On determining probability forecasts from betting
odds"; Shin (1992); Wisdom-of-the-crowd literature on favourite-longshot bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# --------------------------------------------------------------------------- #
# Implied probability / overround
# --------------------------------------------------------------------------- #
def implied_prob(decimal_odds: np.ndarray | pd.Series) -> np.ndarray:
    """Raw implied probability from decimal odds (does NOT sum to 1)."""
    o = np.asarray(decimal_odds, dtype=float)
    return np.where(o > 1.0, 1.0 / o, np.nan)


def booksum(decimal_odds: np.ndarray | pd.Series) -> float:
    """Sum of implied probabilities = 1 + bookmaker margin (the 'overround')."""
    return float(np.nansum(implied_prob(decimal_odds)))


def overround_pct(decimal_odds: np.ndarray | pd.Series) -> float:
    """Bookmaker margin as a percentage, e.g. 18.0 means an 18% book."""
    return (booksum(decimal_odds) - 1.0) * 100.0


def _devig_multiplicative(raw: np.ndarray) -> np.ndarray:
    return raw / np.nansum(raw)


def _devig_power(raw: np.ndarray) -> np.ndarray:
    raw = np.clip(raw, 1e-9, 1 - 1e-9)

    def f(k):
        return np.nansum(raw ** k) - 1.0

    try:
        k = brentq(f, 0.5, 5.0, maxiter=200)
    except ValueError:
        return _devig_multiplicative(raw)
    p = raw ** k
    return p / p.sum()


def _devig_shin(raw: np.ndarray) -> np.ndarray:
    raw = np.clip(raw, 1e-9, 1 - 1e-9)
    B = raw.sum()

    def p_of_z(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * raw ** 2 / B) - z) / (2 * (1 - z))

    def f(z):
        return p_of_z(z).sum() - 1.0

    try:
        z = brentq(f, 1e-6, 0.5, maxiter=200)
    except ValueError:
        return _devig_multiplicative(raw)
    p = p_of_z(z)
    return p / p.sum()


def fair_probabilities(
    decimal_odds: np.ndarray | pd.Series, method: str = "multiplicative"
) -> np.ndarray:
    """Return de-vigged 'fair' market probabilities that sum to 1.

    ``method`` is one of ``multiplicative`` | ``power`` | ``shin``.
    """
    raw = implied_prob(decimal_odds)
    raw = np.where(np.isnan(raw), 0.0, raw)
    if raw.sum() <= 0:
        return raw
    if method == "multiplicative":
        return _devig_multiplicative(raw)
    if method == "power":
        return _devig_power(raw)
    if method == "shin":
        return _devig_shin(raw)
    raise ValueError(f"unknown de-vig method: {method!r}")


def fair_odds(prob: np.ndarray | pd.Series) -> np.ndarray:
    """Fair decimal odds implied by a probability (1/p)."""
    p = np.asarray(prob, dtype=float)
    return np.where(p > 0, 1.0 / p, np.inf)


# --------------------------------------------------------------------------- #
# Value / EV / staking
# --------------------------------------------------------------------------- #
def expected_value(prob: np.ndarray, decimal_odds: np.ndarray) -> np.ndarray:
    """Expected profit per 1 unit staked = p*odds - 1 (a.k.a. the edge).

    Positive => the bet is an *overlay* (model thinks it's underpriced).
    """
    return np.asarray(prob, float) * np.asarray(decimal_odds, float) - 1.0


def kelly_fraction(
    prob: np.ndarray, decimal_odds: np.ndarray, fraction: float = 1.0, cap: float = 0.10
) -> np.ndarray:
    """Kelly stake as a fraction of bankroll.

    f* = (b*p - q) / b, with b = odds-1, q = 1-p. Negative edges -> 0 (no bet).
    ``fraction`` applies fractional Kelly (e.g. 0.25 = quarter-Kelly, the usual
    real-world choice). ``cap`` hard-limits any single stake.
    """
    p = np.asarray(prob, float)
    o = np.asarray(decimal_odds, float)
    b = o - 1.0
    q = 1.0 - p
    f = np.where(b > 0, (b * p - q) / b, 0.0)
    f = np.clip(f, 0.0, None) * fraction
    return np.clip(f, 0.0, cap)


# --------------------------------------------------------------------------- #
# Confidence + risk
# --------------------------------------------------------------------------- #
def confidence_score(
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    field_size: np.ndarray,
    horse_runs: np.ndarray | None = None,
) -> np.ndarray:
    """A transparent 0-100 confidence score for a model selection.

    It rewards (a) a high absolute model probability, (b) the model and market
    broadly agreeing on direction, and (c) larger samples (more career runs,
    not a tiny field). It is intentionally simple and auditable -- it is a
    *ranking aid for humans*, not a second model.
    """
    mp = np.asarray(model_prob, float)
    kp = np.asarray(market_prob, float)
    fs = np.asarray(field_size, float)

    # (a) absolute conviction, scaled by field size (1/N is the uniform baseline).
    baseline = 1.0 / np.clip(fs, 2, None)
    conviction = np.clip((mp - baseline) / (1 - baseline + 1e-9), 0, 1)

    # (b) agreement: penalise wild disagreement with the market (overfitting tell).
    ratio = mp / np.clip(kp, 1e-6, None)
    agreement = np.exp(-np.abs(np.log(np.clip(ratio, 1e-3, 1e3))) / 1.5)

    # (c) sample sufficiency from horse career length.
    if horse_runs is not None:
        hr = np.asarray(horse_runs, float)
        sample = np.clip(hr / 10.0, 0.2, 1.0)
    else:
        sample = np.ones_like(mp)

    score = 100.0 * (0.55 * conviction + 0.30 * agreement + 0.15 * sample)
    return np.clip(score, 0, 100)


def risk_signals(
    model_prob: np.ndarray,
    decimal_odds: np.ndarray,
    fair_market_prob: np.ndarray,
    horse_runs: np.ndarray,
    field_size: np.ndarray,
) -> pd.DataFrame:
    """Per-runner boolean/scalar risk flags for triage before betting."""
    mp = np.asarray(model_prob, float)
    o = np.asarray(decimal_odds, float)
    fmp = np.asarray(fair_market_prob, float)
    hr = np.asarray(horse_runs, float)
    fs = np.asarray(field_size, float)

    ev = expected_value(mp, o)
    return pd.DataFrame({
        # The model thinks it's value but the edge is wafer-thin -> noise risk.
        "thin_edge": (ev > 0) & (ev < 0.05),
        # Big longshot: high variance, sensitive to odds error.
        "longshot": o >= 15.0,
        # Sparse history: the model is extrapolating.
        "low_history": hr < 4,
        # Model wildly disagrees with the market (>2.5x) -> overfitting / stale odds.
        "model_market_divergence": (mp / np.clip(fmp, 1e-6, None)) > 2.5,
        # Huge field: harder to predict, lower per-runner signal.
        "large_field": fs >= 14,
        # The raw edge size, for sorting.
        "edge": ev,
    })


# --------------------------------------------------------------------------- #
# One-call race pricing
# --------------------------------------------------------------------------- #
def price_race(
    runners: pd.DataFrame,
    model_prob_col: str = "model_prob",
    odds_col: str = "win_odds",
    devig_method: str = "multiplicative",
    kelly_fraction_mult: float = 0.25,
) -> pd.DataFrame:
    """Annotate one race's runners with the full probability/value picture.

    Expects a DataFrame of runners for a *single* race containing a model
    probability column and a decimal-odds column. Returns a copy with implied /
    fair / model probabilities, fair odds, edge, EV, Kelly stake, confidence and
    risk flags attached.
    """
    df = runners.copy()
    odds = df[odds_col].to_numpy()
    mp = df[model_prob_col].to_numpy()
    fs = df["field_size"].to_numpy() if "field_size" in df else np.full(len(df), len(df))
    hr = df["horse_runs"].to_numpy() if "horse_runs" in df else np.full(len(df), np.nan)

    df["implied_prob"] = implied_prob(odds)
    df["fair_market_prob"] = fair_probabilities(odds, method=devig_method)
    df["model_fair_odds"] = fair_odds(mp)
    df["edge"] = expected_value(mp, odds)
    df["ev_per_unit"] = df["edge"]
    df["kelly_stake"] = kelly_fraction(mp, odds, fraction=kelly_fraction_mult)
    df["confidence"] = confidence_score(mp, df["fair_market_prob"].to_numpy(), fs, hr)
    df["is_value"] = df["edge"] > 0

    risks = risk_signals(mp, odds, df["fair_market_prob"].to_numpy(), hr, fs)
    risks.index = df.index
    df = pd.concat([df, risks.drop(columns=["edge"])], axis=1)
    return df.sort_values("edge", ascending=False)

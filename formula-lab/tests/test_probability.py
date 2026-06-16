"""Tests for the probability / betting math.

These check mathematical invariants that must always hold regardless of input:
de-vigged probabilities sum to 1, expected value and Kelly behave correctly,
and the de-vig methods move probabilities in sensible directions.
"""
import numpy as np
import pytest

from src.probability.probability import (
    implied_prob,
    booksum,
    overround_pct,
    fair_probabilities,
    fair_odds,
    expected_value,
    kelly_fraction,
    confidence_score,
)

OVERROUND_BOOK = np.array([1.8, 3.5, 5.0, 8.0, 15.0, 26.0])  # booksum > 1


def test_implied_prob_is_reciprocal():
    odds = np.array([2.0, 4.0, 10.0])
    assert np.allclose(implied_prob(odds), [0.5, 0.25, 0.1])


def test_overround_positive_for_real_book():
    assert booksum(OVERROUND_BOOK) > 1.0
    assert overround_pct(OVERROUND_BOOK) > 0.0


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_fair_probabilities_sum_to_one(method):
    p = fair_probabilities(OVERROUND_BOOK, method=method)
    assert np.all(p > 0)
    assert p == pytest.approx(p, rel=0)  # no NaNs
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-6)


def test_fair_odds_roundtrip():
    p = np.array([0.5, 0.3, 0.2])
    assert np.allclose(fair_odds(p), 1.0 / p)


def test_shin_corrects_favourite_longshot_bias():
    # Shin removes proportionally more vig from longshots, so versus naive
    # multiplicative de-vig it RAISES the favourite and LOWERS the longshot.
    mult = fair_probabilities(OVERROUND_BOOK, method="multiplicative")
    shin = fair_probabilities(OVERROUND_BOOK, method="shin")
    assert shin[0] >= mult[0]      # favourite not shrunk
    assert shin[-1] <= mult[-1]    # longshot not inflated


def test_expected_value_sign():
    # Fair coin priced at 2.5 with true p=0.5 -> positive edge.
    assert expected_value(np.array([0.5]), np.array([2.5]))[0] == pytest.approx(0.25)
    # Overpriced favourite -> negative edge.
    assert expected_value(np.array([0.4]), np.array([2.0]))[0] == pytest.approx(-0.2)


def test_kelly_zero_when_no_edge():
    # No edge -> stake nothing (clamped at 0, never negative).
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert kelly_fraction(0.1, 2.0) == 0.0


def test_kelly_positive_and_capped():
    f = kelly_fraction(0.6, 3.0)
    assert 0.0 < f <= 0.10  # positive but respects the safety cap


def test_confidence_score_in_range():
    s = float(confidence_score(model_prob=0.4, market_prob=0.35, field_size=10, horse_runs=20))
    assert 0.0 <= s <= 100.0

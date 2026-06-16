"""
Backtesting and strategy comparison.

Two layers of evaluation:

1. **Model quality** (does it rank winners well, are its probabilities honest?):
   top-1 / top-3 hit rate, mean reciprocal rank, log-loss, Brier score, and a
   calibration table (predicted prob vs realised win frequency).

2. **Strategy quality** (would betting it have made money?): a flat/Kelly bankroll
   simulation over the *out-of-sample* races, reporting hit rate, turnover, P&L,
   ROI / yield, and maximum drawdown -- plus a market-favourite baseline so you
   can see whether the model beats "just back the jolly".

Everything runs on a strict **chronological split**: train on the past, test on
the future. Random splits leak the future into the past and are the classic way
to produce a backtest that never survives contact with a live bankroll.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import schema
from ..features.build_features import FeatureBundle


def time_split(bundle: FeatureBundle, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split by race date. Returns (train_idx, test_idx)."""
    order = bundle.date.sort_values().index
    cut = int(len(order) * train_frac)
    cut_date = bundle.date.loc[order[cut]]
    train_idx = bundle.date.index[bundle.date < cut_date].to_numpy()
    test_idx = bundle.date.index[bundle.date >= cut_date].to_numpy()
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
# Model-quality metrics
# --------------------------------------------------------------------------- #
def ranking_metrics(frame: pd.DataFrame, prob_col: str = "model_prob") -> dict:
    """Top-k hit rate + MRR over races, using the per-race probability ranking."""
    top1 = top3 = mrr = n = 0
    for _, g in frame.groupby(schema.RACE_KEY):
        if g[prob_col].isna().all() or g["won"].sum() == 0:
            continue
        n += 1
        order = g.sort_values(prob_col, ascending=False).reset_index(drop=True)
        win_rank = order.index[order["won"] == 1]
        if len(win_rank) == 0:
            continue
        r = int(win_rank[0]) + 1
        top1 += r == 1
        top3 += r <= 3
        mrr += 1.0 / r
    return {
        "races_scored": n,
        "top1_hit_rate": top1 / n if n else float("nan"),
        "top3_hit_rate": top3 / n if n else float("nan"),
        "mean_reciprocal_rank": mrr / n if n else float("nan"),
    }


def probability_metrics(frame: pd.DataFrame, prob_col: str = "model_prob") -> dict:
    """Log-loss and Brier score on the binary win label."""
    p = frame[prob_col].clip(1e-9, 1 - 1e-9).to_numpy()
    y = frame["won"].fillna(0).to_numpy()
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    return {"log_loss": float(logloss), "brier_score": float(brier)}


def calibration_table(frame: pd.DataFrame, prob_col: str = "model_prob", bins: int = 10) -> pd.DataFrame:
    """Predicted-probability vs realised-win-rate, bucketed. Closer = better."""
    df = frame[[prob_col, "won"]].dropna().copy()
    df["bucket"] = pd.qcut(df[prob_col], q=bins, duplicates="drop")
    tab = df.groupby("bucket", observed=True).agg(
        n=("won", "size"),
        predicted=(prob_col, "mean"),
        actual=("won", "mean"),
    ).reset_index()
    tab["gap"] = tab["actual"] - tab["predicted"]
    return tab


# --------------------------------------------------------------------------- #
# Strategy simulation
# --------------------------------------------------------------------------- #
@dataclass
class StrategyResult:
    name: str
    n_bets: int
    n_wins: int
    hit_rate: float
    total_staked: float
    total_return: float
    profit: float
    roi: float                 # profit / staked  (a.k.a. yield)
    max_drawdown: float
    equity_curve: list[float] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "strategy": self.name,
            "bets": self.n_bets,
            "wins": self.n_wins,
            "hit_rate": round(self.hit_rate, 4),
            "staked": round(self.total_staked, 2),
            "returned": round(self.total_return, 2),
            "profit": round(self.profit, 2),
            "roi_pct": round(self.roi * 100, 2),
            "max_drawdown": round(self.max_drawdown, 2),
        }


def _simulate(bets: pd.DataFrame, name: str, odds_col: str = "win_odds") -> StrategyResult:
    """bets: one row per placed bet with columns [stake, won, odds_col]."""
    if bets.empty:
        return StrategyResult(name, 0, 0, float("nan"), 0, 0, 0, float("nan"), 0, [])
    stake = bets["stake"].to_numpy()
    won = bets["won"].fillna(0).to_numpy()
    odds = bets[odds_col].to_numpy()
    ret = np.where(won == 1, stake * odds, 0.0)      # gross return on the bet
    profit = ret - stake
    equity = np.cumsum(profit)
    peak = np.maximum.accumulate(equity)
    drawdown = float(np.max(peak - equity)) if len(equity) else 0.0
    staked = float(stake.sum())
    return StrategyResult(
        name=name,
        n_bets=int(len(bets)),
        n_wins=int(won.sum()),
        hit_rate=float(won.mean()),
        total_staked=staked,
        total_return=float(ret.sum()),
        profit=float(profit.sum()),
        roi=float(profit.sum() / staked) if staked else float("nan"),
        max_drawdown=drawdown,
        equity_curve=equity.tolist(),
    )


def run_strategies(
    priced: pd.DataFrame,
    *,
    odds_col: str = "win_odds",
    edge_threshold: float = 0.05,
    flat_stake: float = 1.0,
    bankroll: float = 1000.0,
) -> dict[str, StrategyResult]:
    """Simulate several strategies on an already-priced out-of-sample frame.

    ``priced`` must contain, per runner: ``model_prob``, ``edge``, ``win_odds``,
    ``won``, ``kelly_stake`` and the race key. Returns a dict of results.
    """
    results: dict[str, StrategyResult] = {}

    # 1) Model top pick, flat stake (one bet/race on the highest model prob).
    top = (
        priced.sort_values("model_prob", ascending=False)
        .groupby(schema.RACE_KEY, as_index=False)
        .first()
    )
    top = top.assign(stake=flat_stake)
    results["model_top_flat"] = _simulate(top, "model_top_flat", odds_col)

    # 2) Value bets, flat stake (every runner whose edge clears the threshold).
    value = priced[priced["edge"] >= edge_threshold].assign(stake=flat_stake)
    results["value_flat"] = _simulate(value, "value_flat", odds_col)

    # 3) Value bets, fractional-Kelly stake (scaled to bankroll).
    valuek = priced[priced["edge"] >= edge_threshold].copy()
    valuek["stake"] = (valuek["kelly_stake"] * bankroll).clip(lower=0)
    valuek = valuek[valuek["stake"] > 0]
    results["value_kelly"] = _simulate(valuek, "value_kelly", odds_col)

    # 4) Baseline: back the market favourite (shortest price), flat stake.
    fav = (
        priced.sort_values(odds_col, ascending=True)
        .groupby(schema.RACE_KEY, as_index=False)
        .first()
        .assign(stake=flat_stake)
    )
    results["market_favourite"] = _simulate(fav, "market_favourite", odds_col)

    return results


def compare_strategies(results: dict[str, StrategyResult]) -> pd.DataFrame:
    """Tidy comparison table across strategies."""
    return pd.DataFrame([r.as_row() for r in results.values()])

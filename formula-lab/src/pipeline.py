"""
End-to-end orchestration: data -> features -> model -> probabilities -> backtest.

``run_pipeline`` ties the modules together and returns a :class:`PipelineResult`
with the trained model, the priced out-of-sample predictions, and a full set of
evaluation metrics. ``scripts/run_pipeline.py`` is the CLI wrapper around this.

The flow is deliberately linear and inspectable -- each stage hands a typed
object to the next, so you can stop at any point in a notebook and look at the
intermediate frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import backtest as bt
from .data.loader import Dataset, load_dataset
from .features.build_features import FeatureBundle, build_features
from .models.ranker import RaceRanker, RankerConfig
from .models.scoring import ScoringConfig, score_runners
from .probability import probability as prob


@dataclass
class PipelineResult:
    dataset: Dataset
    bundle: FeatureBundle
    ranker: RaceRanker
    priced_test: pd.DataFrame
    metrics: dict
    strategies: dict

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Data source        : {self.dataset.source}",
            f"Races / runs       : {self.dataset.n_races} / {self.dataset.n_runs}",
            f"Model backend      : {self.ranker.backend}",
            f"Top-1 hit rate     : {m['ranking']['top1_hit_rate']:.3f}",
            f"Top-3 hit rate     : {m['ranking']['top3_hit_rate']:.3f}",
            f"Mean recip. rank   : {m['ranking']['mean_reciprocal_rank']:.3f}",
            f"Log loss / Brier   : {m['probability']['log_loss']:.4f} / "
            f"{m['probability']['brier_score']:.4f}",
            "",
            "Strategy comparison (out-of-sample):",
            bt.compare_strategies(self.strategies).to_string(index=False),
        ]
        return "\n".join(lines)


def run_pipeline(
    raw_dir: str = "data/raw",
    *,
    objective: str = "ranking",
    include_market: bool = False,
    devig_method: str = "multiplicative",
    edge_threshold: float = 0.05,
    train_frac: float = 0.7,
    synthetic_kwargs: dict | None = None,
) -> PipelineResult:
    # 1) Load + clean
    ds = load_dataset(raw_dir, synthetic_kwargs=synthetic_kwargs)

    # 2) Features (leakage-safe)
    bundle = build_features(ds.merged, include_market=include_market)

    # 3) Chronological split + train
    train_idx, test_idx = bt.time_split(bundle, train_frac=train_frac)
    ranker = RaceRanker(RankerConfig(objective=objective)).fit(bundle, train_idx)

    # 4) Predict on the held-out future
    model_prob = ranker.predict_proba(bundle, test_idx)

    # 5) Rule-based scores (for explanations / sanity check) on the test rows
    scored = score_runners(bundle.frame.iloc[test_idx], ScoringConfig())

    test_frame = bundle.frame.iloc[test_idx].copy()
    test_frame["model_prob"] = model_prob.values
    for col in ("rule_score", "rule_prob"):
        test_frame[col] = scored[col].values
    for c in scored.columns:
        if str(c).startswith("contrib_"):
            test_frame[c] = scored[c].values

    # 6) Price every race (implied/fair/edge/EV/Kelly/confidence/risk)
    priced_parts = []
    for _, g in test_frame.groupby("race_id"):
        priced_parts.append(
            prob.price_race(g, devig_method=devig_method)
        )
    priced_test = pd.concat(priced_parts).sort_values(["date", "race_id"])

    # 7) Evaluate model + strategies
    metrics = {
        "ranking": bt.ranking_metrics(priced_test),
        "probability": bt.probability_metrics(priced_test),
        "calibration": bt.calibration_table(priced_test),
    }
    strategies = bt.run_strategies(
        priced_test, edge_threshold=edge_threshold
    )

    return PipelineResult(ds, bundle, ranker, priced_test, metrics, strategies)

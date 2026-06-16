# Backtesting

`src/backtest/backtest.py` evaluates both **predictive quality** (is the model
right?) and **economic value** (would acting on it have made money?). These are
different questions and the lab keeps them separate.

## The split must respect time

`time_split(bundle, train_frac=0.7)` sorts by date and trains on the **earliest
70%** of races, tests on the **most recent 30%**. This is non-negotiable for
racing: a random split lets the model learn from the future, which inflates
every metric and produces a backtest that cannot be reproduced live. A race
also never spans both sides of the split (the test enforces this).

## Predictive metrics

- **`ranking_metrics`**: top-1 hit rate (how often the model's top pick wins),
  top-3 hit rate, and mean reciprocal rank (MRR) of the actual winner.
- **`probability_metrics`**: log loss and Brier score on the win probabilities
  — these reward *calibration*, not just picking the winner.
- **`calibration_table`**: buckets predictions and compares predicted vs.
  observed win frequency. A well-calibrated model's 20%-probability horses win
  about 20% of the time. Calibration matters more than accuracy for betting,
  because EV depends on the probability being *right*, not just *ranked* right.

## Strategy simulation

`run_strategies(priced, odds_col, edge_threshold, flat_stake, bankroll)`
backtests four strategies on the out-of-sample races so they can be compared
on equal footing:

1. **`model_top_flat`** — back the model's top pick in every race, flat stake.
2. **`value_flat`** — back every runner whose edge exceeds `edge_threshold`,
   flat stake.
3. **`value_kelly`** — the same value bets, staked by fractional Kelly on a
   fixed notional bankroll (not compounding, so the comparison is clean).
4. **`market_favourite`** — back the market favourite every race. This is the
   **baseline that matters**: a naive bettor paying the takeout. Beating *this*
   is the bar.

Each produces a `StrategyResult` (bets, wins, hit rate, amount staked, returned,
profit, ROI %, max drawdown). `compare_strategies()` stacks them into one table.

## How to read the results honestly

- The market-favourite baseline should lose roughly the **takeout** (~17–22% on
  win pools, less on favourites). If it doesn't, suspect a bug or leakage.
- A strategy that "wins" by a huge margin in backtest is a **red flag**, not a
  triumph — it usually means leakage, an unrealistic price assumption, or
  overfitting to a particular period.
- ROI is noisy. A few hundred bets is not enough to distinguish skill from luck;
  confidence intervals on ROI at realistic sample sizes are wide.
- On the **synthetic** data shipped with this repo the numbers are *illustrative
  of the machinery*, not evidence of profitability — see
  [ASSUMPTIONS_LIMITATIONS.md](ASSUMPTIONS_LIMITATIONS.md).

## Suggested additions

- Bootstrap / block-bootstrap confidence intervals on ROI.
- Walk-forward (rolling-origin) evaluation instead of a single split.
- Commission/rebate and minimum-bet modelling.
- Closing-line value (CLV) tracking — beating the closing price is a better
  long-run signal of edge than short-run ROI.

# 🐎 Horse Race Formula Lab

A clean, **leakage-safe** research framework for horse-racing prediction: from
raw race data → engineered features → model scoring → fair-odds probability →
time-aware backtesting → plain-language LLM explanations.

> ⚠️ **Read this first:** this is a *learning and research* project, not a
> profit system. Real betting markets are highly efficient and carry a large
> takeout (~17–22%). The numbers produced on the bundled synthetic data show
> that the *machinery* works — they are **not** evidence of profitability. See
> [`docs/ASSUMPTIONS_LIMITATIONS.md`](docs/ASSUMPTIONS_LIMITATIONS.md).

## What it does

- **Reviews race data structure** and enforces a strict pre-race vs. result
  column split so post-race information can never leak into features.
- **Two scoring systems**: a transparent glass-box rule scorer and a learned
  LightGBM LambdaRank / classifier — both emit within-race win probabilities.
- **Probability logic**: implied probability, de-vigging (multiplicative /
  power / Shin), expected value/edge, fractional Kelly staking, a transparent
  confidence score, and per-runner risk flags.
- **Backtesting**: chronological train/test split, ranking + calibration
  metrics, and four comparable betting strategies (incl. a market-favourite
  baseline).
- **AI explanations**: fact-grounded race summaries, horse comparisons, and
  prediction reasoning — with deterministic fallbacks when no LLM key is set.
- **LLM/RAG**: race "cards", TF-IDF or dense embeddings, cosine retrieval, and
  a rule-routed race assistant.
- **Honest docs**: assumptions, limitations, and improvement ideas throughout.

## Quickstart

```bash
# 1. Install (LightGBM recommended; falls back to sklearn if absent)
pip install -r requirements.txt

# 2. Run the whole pipeline (uses synthetic data if no real CSVs present)
python scripts/run_pipeline.py

# or drive it from the config file
python scripts/run_pipeline.py --config config/config.yaml

# try a different objective / de-vig method / value threshold
python scripts/run_pipeline.py --objective classification --devig shin --edge 0.07
```

Example output (synthetic data — illustrative only):

```
Data source        : synthetic
Races / runs       : 600 / 6634
Model backend      : lightgbm-lambdarank
Top-1 hit rate     : 0.389
Top-3 hit rate     : 0.735
Mean recip. rank   : 0.599
Log loss / Brier   : 0.2239 / 0.0661

Strategy comparison (out-of-sample):
        strategy  bets  wins  hit_rate  ...  roi_pct  max_drawdown
  model_top_flat   185    72    0.3892  ...    18.65         21.40
      value_flat   303    68    0.2244  ...   -19.97         78.60
     value_kelly   303    68    0.2244  ...    -1.96       1008.76
market_favourite   185    76    0.4108  ...   -10.32         24.10
```

The `market_favourite` baseline losing roughly the takeout is the realistic
floor; treat any strategy that beats it on synthetic data as a demonstration of
the code, not a forecast.

## Using real data

Drop the Kaggle [gdaley/hkracing](https://www.kaggle.com/datasets/gdaley/hkracing)
CSVs into `data/raw/`:

```
data/raw/races.csv
data/raw/runs.csv
```

…either via `python scripts/download_kaggle.py` (needs a Kaggle API token) or
by downloading them manually. The loader picks them up automatically — no code
change. See [`data/README.md`](data/README.md).

## Enabling the LLM (optional)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...   # uses claude-sonnet-4-6
```

Without a key, all explanation/assistant features still work using deterministic
templates and local TF-IDF retrieval. See
[`docs/LLM_FEATURES.md`](docs/LLM_FEATURES.md).

## Project layout

```
config/            config.yaml — central settings
data/raw/          drop real races.csv + runs.csv here (git-ignored)
src/
  data/            schema.py (column truth), loader.py, synthetic.py
  features/        build_features.py  (LEAKAGE-SAFE feature engineering)
  models/          scoring.py (rule scorer), ranker.py (LightGBM)
  probability/     probability.py (de-vig, EV, Kelly, confidence, risk)
  backtest/        backtest.py (time split, metrics, strategies)
  llm/             explain.py, rag.py, assistant.py
  pipeline.py      end-to-end orchestration
scripts/           run_pipeline.py, download_kaggle.py
tests/             test_probability.py, test_leakage.py
docs/              DATA_STRUCTURE, MODEL_DESIGN, PROBABILITY_LOGIC,
                   BACKTESTING, LLM_FEATURES, ASSUMPTIONS_LIMITATIONS
```

## Design principles

1. **No leakage, ever** — prior-only features, strict column split,
   chronological backtest. Enforced by tests.
2. **Model independent of the market by default** — so "value" means the model
   genuinely disagrees with the de-vigged price, not that it imitates it.
3. **Glass-box where it counts** — auditable scoring, transparent confidence,
   fact-grounded explanations.
4. **Runs anywhere** — graceful fallbacks: synthetic data, sklearn without
   LightGBM, template text without an LLM key, TF-IDF without dense embeddings.
5. **Honesty over hype** — every doc states what the system can and cannot do.

## Documentation

| Doc | Contents |
|-----|----------|
| [DATA_STRUCTURE.md](docs/DATA_STRUCTURE.md) | dataset schema, pre-race vs. result columns |
| [MODEL_DESIGN.md](docs/MODEL_DESIGN.md) | features, rule scorer, learned ranker |
| [PROBABILITY_LOGIC.md](docs/PROBABILITY_LOGIC.md) | de-vigging, EV, Kelly, confidence, risk |
| [BACKTESTING.md](docs/BACKTESTING.md) | splits, metrics, strategies, how to read results |
| [LLM_FEATURES.md](docs/LLM_FEATURES.md) | explanations, RAG, embeddings, assistant |
| [ASSUMPTIONS_LIMITATIONS.md](docs/ASSUMPTIONS_LIMITATIONS.md) | caveats, responsible use |

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE). For education and research. Please gamble
responsibly; never bet money you cannot afford to lose.

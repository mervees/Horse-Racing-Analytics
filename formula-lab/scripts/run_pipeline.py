#!/usr/bin/env python
"""Run the end-to-end Horse Race Formula Lab pipeline.

Examples
--------
# Use defaults (synthetic data if no real CSVs in data/raw):
python scripts/run_pipeline.py

# Load settings from the YAML config:
python scripts/run_pipeline.py --config config/config.yaml

# Override a few knobs on the command line:
python scripts/run_pipeline.py --objective classification --devig shin --edge 0.07

# Save the trained model and write a markdown report:
python scripts/run_pipeline.py --save-model models/ranker.joblib --report out/report.md
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `src` importable when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_pipeline  # noqa: E402
from src.backtest.backtest import compare_strategies  # noqa: E402


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_config(args: argparse.Namespace) -> dict:
    """Build run_pipeline kwargs from config file + CLI overrides."""
    cfg: dict = {}
    if args.config:
        raw = _load_yaml(args.config)
        cfg["raw_dir"] = raw.get("data", {}).get("raw_dir", "data/raw")
        cfg["synthetic_kwargs"] = raw.get("data", {}).get("synthetic")
        cfg["objective"] = raw.get("model", {}).get("objective", "ranking")
        cfg["include_market"] = raw.get("model", {}).get("include_market", False)
        cfg["devig_method"] = raw.get("probability", {}).get("devig_method", "multiplicative")
        cfg["edge_threshold"] = raw.get("backtest", {}).get("edge_threshold", 0.05)
        cfg["train_frac"] = raw.get("backtest", {}).get("train_frac", 0.7)

    # CLI overrides (only when explicitly provided)
    if args.raw_dir is not None:
        cfg["raw_dir"] = args.raw_dir
    if args.objective is not None:
        cfg["objective"] = args.objective
    if args.include_market:
        cfg["include_market"] = True
    if args.devig is not None:
        cfg["devig_method"] = args.devig
    if args.edge is not None:
        cfg["edge_threshold"] = args.edge
    if args.train_frac is not None:
        cfg["train_frac"] = args.train_frac
    if args.n_races is not None:
        cfg["synthetic_kwargs"] = {**(cfg.get("synthetic_kwargs") or {}), "n_races": args.n_races}

    # drop None values so run_pipeline defaults apply
    return {k: v for k, v in cfg.items() if v is not None}


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Horse Race Formula Lab pipeline.")
    p.add_argument("--config", help="Path to a YAML config (e.g. config/config.yaml).")
    p.add_argument("--raw-dir", dest="raw_dir", help="Directory with races.csv + runs.csv.")
    p.add_argument("--objective", choices=["ranking", "classification"])
    p.add_argument("--include-market", action="store_true",
                   help="Fold market odds into model features (default: kept separate).")
    p.add_argument("--devig", choices=["multiplicative", "power", "shin"])
    p.add_argument("--edge", type=float, help="Minimum model edge for a value bet.")
    p.add_argument("--train-frac", dest="train_frac", type=float)
    p.add_argument("--n-races", dest="n_races", type=int,
                   help="Synthetic race count (only used if falling back to synthetic).")
    p.add_argument("--save-model", dest="save_model", help="Path to save the trained model (.joblib).")
    p.add_argument("--report", help="Path to write a markdown run report.")
    args = p.parse_args()

    kwargs = _merge_config(args)
    print(">> running pipeline with:", kwargs or "(defaults)")
    res = run_pipeline(**kwargs)

    print("\n" + res.summary())

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        res.ranker.save(args.save_model)
        print(f"\nSaved model -> {args.save_model}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        table = compare_strategies(res.strategies)
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("# Pipeline run report\n\n")
            fh.write("```\n" + res.summary() + "\n```\n\n")
            fh.write("## Strategy comparison (out-of-sample)\n\n")
            fh.write(table.to_markdown(index=False) + "\n")
        print(f"Wrote report -> {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

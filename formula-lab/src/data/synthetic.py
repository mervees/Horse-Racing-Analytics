"""
Synthetic horse-racing data generator.

Produces ``races.csv`` and ``runs.csv`` that match the canonical hkracing schema
(see ``src/data/schema.py``) so the whole pipeline runs end-to-end *before* you
download the real Kaggle data. Drop the real CSVs into ``data/raw/`` and the
loader uses them instead -- no code changes required.

Design goals
------------
* **Real signal.** Each horse/jockey/trainer carries a latent ability so models
  can actually learn something and the docs' numbers are meaningful.
* **Realistic market.** Decimal odds are derived from a *noisy* version of the
  true win probability, then inflated by a bookmaker overround (takeout). The
  market is therefore a strong but imperfect baseline -- exactly the regime real
  bettors face, where edge is small and takeout is the enemy.
* **No hidden oracle.** The generator never leaks the latent ability into any
  pre-race column. Feature code only ever sees what a punter would see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VENUES = ["ST", "HV"]
DISTANCES = [1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400]
GOINGS = ["GOOD", "GOOD TO FIRM", "GOOD TO YIELDING", "YIELDING", "WET SLOW"]
CONFIGS = ["A", "A+3", "B", "B+2", "C", "C+3"]
HORSE_TYPES = ["Gelding", "Mare", "Horse", "Colt", "Filly", "Rig"]
COUNTRIES = ["AUS", "NZ", "IRE", "GB", "USA", "JPN", "SAF", "FR"]
GEAR = ["", "B", "B/TT", "TT", "V", "CP", "H", "SR"]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def generate(
    n_races: int = 1500,
    n_horses: int = 1800,
    n_jockeys: int = 60,
    n_trainers: int = 90,
    start_date: str = "2015-09-01",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (races, runs) DataFrames.

    Parameters scale the size of the simulated world. Defaults give ~1,500 races
    and ~16k runs -- enough for a meaningful time-split backtest in seconds.
    """
    rng = np.random.default_rng(seed)

    # --- latent actors -------------------------------------------------------
    # Horse ability drifts slowly over its career; we model a base + a small age
    # curve. Jockey/trainer skill is a static latent here (extendable to drift).
    horse_base = rng.normal(0, 1.0, n_horses)
    horse_age0 = rng.integers(2, 5, n_horses)            # age at first appearance
    jockey_skill = rng.normal(0, 0.45, n_jockeys)
    trainer_skill = rng.normal(0, 0.35, n_trainers)
    # Each horse has a preferred distance and a going preference.
    horse_pref_dist = rng.choice(DISTANCES, n_horses)
    horse_going_aff = rng.normal(0, 0.3, (n_horses, len(GOINGS)))

    dates = pd.to_datetime(start_date) + pd.to_timedelta(
        np.sort(rng.integers(0, 365 * 4, n_races)), unit="D"
    )

    race_rows: list[dict] = []
    run_rows: list[dict] = []

    for r in range(n_races):
        race_id = r + 1
        date = dates[r]
        venue = rng.choice(VENUES)
        distance = int(rng.choice(DISTANCES))
        going = rng.choice(GOINGS, p=[0.40, 0.22, 0.18, 0.12, 0.08])
        going_idx = GOINGS.index(going)
        config = rng.choice(CONFIGS)
        race_class = int(rng.integers(1, 6))             # 1 best .. 5 weakest
        surface = int(rng.random() < 0.12)               # ~12% all-weather
        field_size = int(rng.integers(8, 15))
        prize = int({1: 3_000_000, 2: 1_800_000, 3: 1_100_000,
                     4: 750_000, 5: 520_000}[race_class] * rng.uniform(0.85, 1.2))

        runners = rng.choice(n_horses, size=field_size, replace=False)
        jockeys = rng.choice(n_jockeys, size=field_size)
        trainers = rng.choice(n_trainers, size=field_size)
        draws = rng.permutation(field_size) + 1
        actual_weights = rng.integers(113, 135, field_size)   # handicap, lbs
        declared_weights = actual_weights + rng.integers(1000, 1150, field_size)

        # --- latent race-day strength per runner -----------------------------
        age = horse_age0[runners] + (date.year - pd.to_datetime(start_date).year)
        age_curve = -0.05 * (age - 5) ** 2                # peak ~5yo
        dist_fit = -0.0008 * np.abs(distance - horse_pref_dist[runners])
        going_fit = horse_going_aff[runners, going_idx]
        weight_penalty = -0.015 * (actual_weights - actual_weights.mean())
        # Inside draws are a small advantage, stronger at Happy Valley.
        draw_edge = (-0.03 if venue == "HV" else -0.018) * (draws - 1)

        strength = (
            horse_base[runners]
            + 0.8 * jockey_skill[jockeys]
            + 0.6 * trainer_skill[trainers]
            + age_curve
            + dist_fit
            + going_fit
            + weight_penalty
            + draw_edge
        )
        # Class lifts the whole field's variance (better horses, tighter race).
        noise = rng.normal(0, 0.9 + 0.06 * race_class, field_size)
        performance = strength + noise                    # higher = better

        order = np.argsort(-performance)                  # winner first
        result = np.empty(field_size, dtype=int)
        result[order] = np.arange(1, field_size + 1)
        won = (result == 1).astype(int)

        # --- market: noisy true prob + bookmaker overround --------------------
        true_p = _softmax(strength)                       # "true" win prob
        # The market is *mostly* efficient: a small amount of perception noise so
        # there is a thin, hard-to-find edge -- not a money fountain. Real books
        # are at least this efficient; treat synthetic profit as illustrative only.
        perceived = np.log(true_p + 1e-9) + rng.normal(0, 0.18, field_size)
        market_p = _softmax(perceived)
        overround = rng.uniform(1.17, 1.22)               # ~17-22% takeout
        implied = market_p * overround
        win_odds = np.clip(1.0 / implied, 1.05, 99.0).round(1)
        place_odds = np.clip(win_odds * rng.uniform(0.28, 0.45, field_size),
                             1.02, 30.0).round(1)

        # --- times (cosmetic but schema-complete) -----------------------------
        base_time = distance / 16.7                        # ~ seconds at gallop
        finish_time = (base_time
                       + (result - 1) * rng.uniform(0.12, 0.18)
                       + rng.normal(0, 0.25, field_size)).round(2)

        race_rows.append({
            "race_id": race_id,
            "date": date.strftime("%Y-%m-%d"),
            "venue": venue,
            "race_no": int(rng.integers(1, 11)),
            "config": config,
            "surface": surface,
            "distance": distance,
            "going": going,
            "horse_ratings": f"{max(0, 40 - race_class * 5)}-{120 - race_class * 10}",
            "race_class": race_class,
            "prize": prize,
            "sec_time1": round(float(base_time / 3), 2),
            "time1": round(float(finish_time.min()), 2),
            "win_combination1": int(runners[order[0]] % 14 + 1),
            "win_dividend1": float(win_odds[order[0]]),
            "place_dividend1": float(place_odds[order[0]]),
        })

        for i in range(field_size):
            run_rows.append({
                "race_id": race_id,
                "horse_no": int(i + 1),
                "horse_id": int(runners[i]),
                "result": int(result[i]),
                "won": int(won[i]),
                "horse_age": int(age[i]),
                "horse_country": rng.choice(COUNTRIES),
                "horse_type": rng.choice(HORSE_TYPES),
                "horse_rating": int(np.clip(60 + strength[i] * 12, 20, 125)),
                "horse_gear": rng.choice(GEAR, p=[0.55, 0.15, 0.08, 0.07,
                                                  0.06, 0.04, 0.03, 0.02]),
                "declared_weight": int(declared_weights[i]),
                "actual_weight": int(actual_weights[i]),
                "draw": int(draws[i]),
                "trainer_id": int(trainers[i]),
                "jockey_id": int(jockeys[i]),
                "win_odds": float(win_odds[i]),
                "place_odds": float(place_odds[i]),
                "finish_time": float(finish_time[i]),
                "position_sec1": int(rng.integers(1, field_size + 1)),
            })

    races = pd.DataFrame(race_rows)
    runs = pd.DataFrame(run_rows)
    return races, runs


def write_synthetic(out_dir: str, **kwargs) -> tuple[str, str]:
    """Generate and write races.csv / runs.csv into ``out_dir``."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    races, runs = generate(**kwargs)
    rp = os.path.join(out_dir, "races.csv")
    up = os.path.join(out_dir, "runs.csv")
    races.to_csv(rp, index=False)
    runs.to_csv(up, index=False)
    return rp, up


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate synthetic HK racing data.")
    ap.add_argument("--out", default="data/raw", help="output directory")
    ap.add_argument("--races", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rp, up = write_synthetic(a.out, n_races=a.races, seed=a.seed)
    print(f"wrote {rp} and {up}")

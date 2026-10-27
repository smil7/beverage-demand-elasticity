"""Hyperparameter tuning for LightGBM with time-series cross-validation.

The tuning folds sit *before* the backtest period, so tuning never sees the weeks used to report
accuracy:

    tuning folds:   origins 303, 311, 319  -> validate on weeks 304-327
    backtest folds: origins 327 ... 391    -> report on weeks 328-399

Each candidate setting is trained on everything up to the fold origin and validated on the next
8 weeks (early stopping picks the number of trees). Every trial is logged to the ``tuning_trials``
table in the warehouse and to ``results/tuning_trials.csv``; the winner is written to
``results/best_params.json`` and used by the backtest.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from beverage_demand.config import results, settings
from beverage_demand.db import connect, write_table
from beverage_demand.features.panel import load_panel
from beverage_demand.forecasting.features import build_supervised, feature_columns
from beverage_demand.forecasting.metrics import wape
from beverage_demand.forecasting.models import lgb_params, train_lgb

SEARCH_SPACE = {
    "tweedie_variance_power": [1.2, 1.5, 1.7],
    "num_leaves": [31, 127],
    "min_data_in_leaf": [50, 200],
}
N_TRIALS = 8  # random sample from the 12-point grid


def cv_folds(first_backtest_origin: int, step: int, n: int = 3) -> list[int]:
    return [first_backtest_origin - step * k for k in range(n, 0, -1)]


def sample_grid(seed: int, n: int) -> list[dict]:
    keys = list(SEARCH_SPACE)
    grid = [dict(zip(keys, v, strict=True)) for v in itertools.product(*SEARCH_SPACE.values())]
    rng = np.random.default_rng(seed)
    return [grid[i] for i in sorted(rng.choice(len(grid), size=min(n, len(grid)), replace=False))]


def run() -> dict:
    cfg = settings()
    fc = cfg["forecast"]
    H = fc["horizon"]
    features = feature_columns(fc["lags"], fc["seasonal_lag"])
    sup = build_supervised(load_panel(), cfg["scope"]["brands"], H, fc["lags"], fc["seasonal_lag"])
    sup = sup[(sup["data_gap"] == 0) & sup["units"].notna()].reset_index(drop=True)
    origins = cv_folds(fc["first_origin"], fc["fold_step"])
    print(f"Tuning folds (origins): {origins}")

    trials = []
    for t, cand in enumerate(sample_grid(fc["seed"], N_TRIALS)):
        params = lgb_params(features, fc["seed"], cand)
        scores, rounds = [], []
        t0 = time.time()
        for T in origins:
            train = sup[sup["week"] <= T]
            valid = sup[(sup["origin_week"] == T) & (sup["week"] > T)]
            model = train_lgb(train, features, params, 2000, valid=valid)
            scores.append(wape(valid["units"].to_numpy(), model.predict(valid[features])))
            rounds.append(model.best_iteration)
        trials.append({"trial": t, **cand, **{f"wape_fold{i}": s for i, s in enumerate(scores)},
                       "cv_wape": float(np.mean(scores)), "cv_wape_std": float(np.std(scores)),
                       "rounds": int(np.mean(rounds)), "seconds": round(time.time() - t0)})
        print(f"  trial {t}: {cand} -> CV WAPE {np.mean(scores):.4f} "
              f"(rounds {int(np.mean(rounds))}, {time.time() - t0:.0f}s)")

    log = pd.DataFrame(trials).sort_values("cv_wape").reset_index(drop=True)
    log.to_csv(results("tuning_trials.csv"), index=False)
    with connect() as con:
        write_table(con, "tuning_trials", log.assign(tuned_at=pd.Timestamp.now()))

    best = log.iloc[0]
    out = {"params": {k: log.loc[0, k].item() for k in SEARCH_SPACE},  # per column keeps int types
           "rounds": int(best["rounds"] * 1.1),  # a bit more data in the final fit -> ~10% more trees
           "cv_wape": float(best["cv_wape"]), "cv_folds": origins}
    results("best_params.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Best:", out)
    return out


if __name__ == "__main__":
    run()

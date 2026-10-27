"""Hyperparameter tuning for LightGBM with time-series cross-validation.

The tuning folds sit *before* the backtest period, so tuning never sees the weeks used to report
accuracy:

    tuning folds:   origins 303, 311, 319  -> validate on weeks 304-327
    backtest folds: origins 327 ... 391    -> report on weeks 328-399

Each candidate setting is trained for MAX_ROUNDS trees on everything up to each fold origin and
scored on the next 8 weeks after every tree. The validation curves of the 3 folds are averaged and
the number of trees with the lowest average WAPE is chosen -- more stable than averaging each
fold's own early-stopping point. Trials go to ``results/tuning_trials.csv``; the winner to
``results/best_params.json`` (used by the backtest).
"""

from __future__ import annotations

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from beverage_demand.config import results, settings
from beverage_demand.features.panel import load_panel
from beverage_demand.forecasting.features import CATEGORICAL, build_supervised, feature_columns
from beverage_demand.forecasting.models import lgb_params

SEARCH_SPACE = {
    "tweedie_variance_power": [1.2, 1.5, 1.7],
    "num_leaves": [31, 127],
    "min_data_in_leaf": [50, 200],
}
N_TRIALS = 8  # random sample from the 12-point grid
MAX_ROUNDS = 300


def cv_folds(first_backtest_origin: int, step: int, n: int = 3) -> list[int]:
    return [first_backtest_origin - step * k for k in range(n, 0, -1)]


def sample_grid(seed: int, n: int) -> list[dict]:
    keys = list(SEARCH_SPACE)
    grid = [dict(zip(keys, v, strict=True)) for v in itertools.product(*SEARCH_SPACE.values())]
    rng = np.random.default_rng(seed)
    return [grid[i] for i in sorted(rng.choice(len(grid), size=min(n, len(grid)), replace=False))]


def wape_curve(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], params: dict) -> np.ndarray:
    """Validation WAPE after each tree. LightGBM's l1 metric is the mean absolute error, and
    WAPE = MAE x n / total actual units on a fixed validation set."""
    evals: dict = {}
    dtrain = lgb.Dataset(train[features], train["units"], categorical_feature=CATEGORICAL)
    dvalid = lgb.Dataset(valid[features], valid["units"], reference=dtrain)
    lgb.train(params | {"metric": "l1"}, dtrain, num_boost_round=MAX_ROUNDS,
              valid_sets=[dvalid], valid_names=["valid"], callbacks=[lgb.record_evaluation(evals)])
    return np.array(evals["valid"]["l1"]) * len(valid) / valid["units"].sum()


def run() -> dict:
    cfg = settings()
    fc = cfg["forecast"]
    features = feature_columns(fc["lags"], fc["seasonal_lag"])
    sup = build_supervised(load_panel(), cfg["scope"]["brands"], fc["horizon"], fc["lags"],
                           fc["seasonal_lag"])
    sup = sup[(sup["data_gap"] == 0) & sup["units"].notna()].reset_index(drop=True)
    origins = cv_folds(fc["first_origin"], fc["fold_step"])
    print(f"Tuning folds (origins): {origins}")

    trials = []
    for t, cand in enumerate(sample_grid(fc["seed"], N_TRIALS)):
        t0 = time.time()
        params = lgb_params(features, fc["seed"], cand)
        curves = np.array([
            wape_curve(sup[sup["week"] <= T], sup[(sup["origin_week"] == T) & (sup["week"] > T)],
                       features, params)
            for T in origins])
        mean_curve = curves.mean(axis=0)
        best = int(mean_curve.argmin())
        trials.append({"trial": t, **cand,
                       **{f"wape_fold{i}": c[best] for i, c in enumerate(curves)},
                       "cv_wape": float(mean_curve[best]), "rounds": best + 1,
                       "seconds": round(time.time() - t0)})
        print(f"  trial {t}: {cand} -> CV WAPE {mean_curve[best]:.4f} at {best + 1} trees "
              f"({time.time() - t0:.0f}s)")

    log = pd.DataFrame(trials).sort_values("cv_wape").reset_index(drop=True)
    log.to_csv(results("tuning_trials.csv"), index=False)
    out = {"params": {k: log.loc[0, k].item() for k in SEARCH_SPACE},  # per column keeps int types
           "rounds": int(log.loc[0, "rounds"]),
           "cv_wape": float(log.loc[0, "cv_wape"]), "cv_folds": origins}
    results("best_params.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Best:", out)
    return out


if __name__ == "__main__":
    run()

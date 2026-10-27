"""LightGBM objective selection on the PRE-backtest validation window only.

Target weeks 320-327 forecast from origin 319 -- strictly before the first backtest origin (327),
so choosing the objective here does not peek at the backtest folds.
Usage: uv run python scripts/select_lightgbm_objective.py
"""
import time

import lightgbm as lgb
import numpy as np

from beverage_demand.config import settings
from beverage_demand.features.panel import load_panel
from beverage_demand.forecasting.backtest import lgb_params
from beverage_demand.forecasting.features import CATEGORICAL, build_supervised, feature_columns

cfg = settings()
fc = cfg["forecast"]
feats = feature_columns(fc["lags"], 52)
sup = build_supervised(load_panel(), cfg["scope"]["brands"], 8, fc["lags"])
sup = sup[(sup.data_gap == 0) & sup.units.notna()]
o = fc["first_origin"] - 8
tr = sup[sup.week <= o]
va = sup[(sup.origin_week == o) & (sup.week > o)]
y = va.units.to_numpy()

def run(name, upd, target="raw"):
    p = lgb_params(42, feats) | upd
    ytr = tr.units if target == "raw" else np.log1p(tr.units)
    t = time.time()
    m = lgb.train(p, lgb.Dataset(tr[feats], ytr, categorical_feature=CATEGORICAL), 1500,
                  valid_sets=[lgb.Dataset(va[feats], y if target == "raw" else np.log1p(y))],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    pr = m.predict(va[feats])
    pr = pr if target == "raw" else np.expm1(pr)
    w = np.abs(y - pr).sum() / y.sum()
    print(f"{name:<40} rounds {m.best_iteration:>4}  WAPE {w:.4f}  ({time.time()-t:.0f}s)")

run("tweedie 1.2", {"tweedie_variance_power": 1.2})
run("tweedie 1.5", {"tweedie_variance_power": 1.5})
run("tweedie 1.7 (selected)", {"tweedie_variance_power": 1.7})
run("poisson", {"objective": "poisson"})
run("l2 on log1p", {"objective": "regression"}, target="log")

"""Model definitions and training helpers for the demand forecast.

Two machine-learning models are compared:

* **LightGBM** (main model) -- gradient-boosted trees, Tweedie loss, tuned with time-series CV
  (``forecasting.tuning``).
* **HistGradientBoosting** (scikit-learn) -- a second tree-boosting implementation with a Poisson
  loss, used as a comparison model with fixed, sensible settings.

Both use monotone constraints: predicted volume may not go *up* when our own price goes up, or
*down* when the discount gets deeper. This keeps the price response sensible, which matters
because the promotion analysis asks the model "what if there had been no promotion?".
"""

from __future__ import annotations

import json

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from beverage_demand.config import results
from beverage_demand.features.panel import PROMO_TYPES
from beverage_demand.forecasting.features import CATEGORICAL

MONOTONE = {"ln_price": -1, "discount": 1, "price_vs_comp": -1}

# starting point before tuning; tuning.py searches around these values
DEFAULT_LGB = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
}


def lgb_params(features: list[str], seed: int, overrides: dict | None = None) -> dict:
    return DEFAULT_LGB | (overrides or {}) | {
        "monotone_constraints": [MONOTONE.get(f, 0) for f in features],
        "monotone_constraints_method": "advanced",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
        "num_threads": 0,
    }


def train_lgb(train: pd.DataFrame, features: list[str], params: dict, rounds: int,
              valid: pd.DataFrame | None = None) -> lgb.Booster:
    """Train LightGBM; with ``valid`` given, stop early when validation error stops improving."""
    dtrain = lgb.Dataset(train[features], train["units"], categorical_feature=CATEGORICAL)
    if valid is None:
        return lgb.train(params, dtrain, num_boost_round=rounds)
    dvalid = lgb.Dataset(valid[features], valid["units"], reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def train_hgb(train: pd.DataFrame, features: list[str], seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.1,
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        categorical_features=[features.index(c) for c in CATEGORICAL],
        monotonic_cst=[MONOTONE.get(f, 0) for f in features],
        early_stopping=False,
        random_state=seed,
    )
    return model.fit(train[features], train["units"])


def load_tuned_lgb(features: list[str], seed: int) -> tuple[dict, int]:
    """Best LightGBM settings from ``results/best_params.json`` (written by tuning.py)."""
    path = results("best_params.json")
    if not path.exists():
        raise FileNotFoundError("Run `uv run bev tune` first to choose the LightGBM settings")
    best = json.loads(path.read_text(encoding="utf-8"))
    return lgb_params(features, seed, best["params"]), int(best["rounds"])


def counterfactual(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows with the promotion removed: regular price, no discount, no promo flags."""
    cf = df.copy()
    delta = cf["ln_reg_price"] - cf["ln_price"]
    cf["ln_price"] = cf["ln_reg_price"]
    cf["price_vs_comp"] = cf["price_vs_comp"] + delta
    cf["discount"] = 0.0
    for t in PROMO_TYPES:
        cf[f"promo_{t.lower()}"] = 0
    return cf

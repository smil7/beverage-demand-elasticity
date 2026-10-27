"""Phase 4: rolling-origin backtest of LightGBM vs. baselines (no random splits).

For each fold with origin ``T`` the models are trained on target weeks ``<= T`` and forecast weeks
``T+1 .. T+H``, each week at its true horizon ``h = t - T``. Baselines:

* ``seasonal_naive`` -- same week last year (falls back to the 4-week mean if missing)
* ``moving_avg_4``   -- mean of the last 4 observed weeks before the origin
* ``regression``     -- the Phase 3 log-log model, re-fitted on each fold's training window
                        (linear trend instead of year dummies so it can extrapolate)
* ``lightgbm``       -- global gradient-boosted model over all series and stores

The fold models also produce the *promo counterfactual* used in Phase 5: the same week predicted
with promotion switched off and price set back to the regular price.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from beverage_demand.config import processed, results, settings
from beverage_demand.elasticity.model import estimation_sample, fit_series
from beverage_demand.features.panel import PROMO_TYPES, load_panel
from beverage_demand.forecasting.features import CATEGORICAL, build_supervised, feature_columns
from beverage_demand.forecasting.metrics import score

MODELS = ["lightgbm", "seasonal_naive", "moving_avg_4", "regression"]
BASELINES = ["seasonal_naive", "moving_avg_4", "regression"]


def lgb_params(seed: int, features: list[str]) -> dict:
    # Monotone constraints keep the price response economically sensible (needed for the promo
    # counterfactual): higher own price -> lower volume, deeper discount -> higher volume.
    mono = {"ln_price": -1, "discount": 1, "price_vs_comp": -1}
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.7,  # chosen in scripts/select_lightgbm_objective.py
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "monotone_constraints": [mono.get(f, 0) for f in features],
        "monotone_constraints_method": "advanced",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
        "num_threads": 0,
    }


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


def train_lgb(train: pd.DataFrame, features: list[str], params: dict, rounds: int,
              valid: pd.DataFrame | None = None) -> lgb.Booster:
    dtrain = lgb.Dataset(train[features], train["units"], categorical_feature=CATEGORICAL,
                         free_raw_data=True)
    if valid is None:
        return lgb.train(params, dtrain, num_boost_round=rounds)
    dvalid = lgb.Dataset(valid[features], valid["units"], reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def regression_forecast(panel: pd.DataFrame, origin: int, horizon: int, brands: list[str]) -> pd.DataFrame:
    """Fit the log-log model per series on weeks <= origin; predict weeks origin+1..origin+H."""
    est = estimation_sample(panel[panel["week"] <= origin])
    test = panel[(panel["week"] > origin) & (panel["week"] <= origin + horizon)]
    out = []
    for series, d in est.groupby("series"):
        m = fit_series(d, d["brand"].iloc[0], brands, time_effects="trend")
        t = test[test["series"] == series]
        out.append(t[["series", "store", "week"]].assign(regression=m.predict_units(t)))
    return pd.concat(out, ignore_index=True)


def run() -> dict:
    cfg = settings()
    fc = cfg["forecast"]
    brands = cfg["scope"]["brands"]
    H, lags = fc["horizon"], fc["lags"]
    features = feature_columns(lags, fc["seasonal_lag"])
    params = lgb_params(fc["seed"], features)
    t0 = time.time()

    panel = load_panel()
    sup = build_supervised(panel, brands, H, lags, fc["seasonal_lag"])
    sup = sup[(sup["data_gap"] == 0) & sup["units"].notna()].reset_index(drop=True)
    print(f"Supervised frame: {len(sup):,} rows x {len(features)} features ({time.time() - t0:.0f}s)")

    origins = [fc["first_origin"] + i * fc["fold_step"] for i in range(fc["n_folds"])]

    # number of boosting rounds: early stopping on the 8 weeks before the first origin only
    o0 = origins[0]
    tr = sup[sup["week"] <= o0 - H]
    va = sup[(sup["week"] > o0 - H) & (sup["week"] <= o0) & (sup["origin_week"] == o0 - H)]
    probe = train_lgb(tr, features, params, 2000, valid=va)
    rounds = int(probe.best_iteration * 1.1)
    print(f"Boosting rounds chosen on pre-backtest validation: {rounds}")

    preds, cf_rows, last_model, last_test = [], [], None, None
    for i, T in enumerate(origins):
        tf = time.time()
        train = sup[sup["week"] <= T]
        model = train_lgb(train, features, params, rounds)

        # forecasts made at origin T for weeks T+1..T+H (each at its own horizon)
        test = sup[(sup["origin_week"] == T) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
        test["fold"] = i
        test["lightgbm"] = model.predict(test[features])
        test["moving_avg_4"] = test["rmean_4"]
        test["seasonal_naive"] = test["lag_52"].fillna(test["rmean_4"])
        reg = regression_forecast(panel, T, H, brands)
        test = test.merge(reg, on=["series", "store", "week"], how="left")
        test["regression"] = test["regression"].fillna(test["rmean_4"])
        preds.append(test)

        # promo counterfactual on 1-week-ahead rows of the same test weeks
        now = sup[(sup["h"] == 1) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
        now["fold"] = i
        now["pred_actual"] = model.predict(now[features])
        now["pred_baseline"] = model.predict(counterfactual(now)[features])
        cf_rows.append(now)
        last_model, last_test = model, test
        print(f"  fold {i}: origin week {T}, train rows {len(train):,}, "
              f"WAPE lgbm {np.abs(test.units - test.lightgbm).sum() / test.units.sum():.3f} "
              f"({time.time() - tf:.0f}s)")

    pred = pd.concat(preds, ignore_index=True)
    keep = ["series", "brand", "pack", "store", "week", "h", "fold", "units", "promo",
            "promo_type", "discount", *MODELS]
    pred[keep].to_parquet(processed("backtest_predictions"), index=False)
    cf = pd.concat(cf_rows, ignore_index=True)
    cf_keep = ["series", "brand", "pack", "store", "week", "fold", "units", "promo", "promo_type",
               "discount", "unit_price", "reg_price", "unit_cost", "pred_actual", "pred_baseline"]
    cf[cf_keep].to_parquet(processed("promo_counterfactual"), index=False)
    last_model.save_model(str(processed("lightgbm_last_fold").with_suffix(".txt")))
    last_test[features].sample(min(5000, len(last_test)), random_state=0).to_parquet(
        processed("shap_sample"), index=False)

    pred["segment"] = np.where(pred["promo"] == 1, "promo weeks", "non-promo weeks")
    metrics = pd.concat([
        score(pred, MODELS).assign(slice="overall", group="all"),
        score(pred, MODELS, by=["segment"]).rename(columns={"segment": "group"}).assign(slice="promo"),
        score(pred, MODELS, by=["series"]).rename(columns={"series": "group"}).assign(slice="series"),
        score(pred, MODELS, by=["h"]).rename(columns={"h": "group"}).assign(slice="horizon"),
        score(pred, MODELS, by=["series", "segment"]).assign(
            group=lambda d: d["series"] + " / " + d["segment"], slice="series_promo"
        ).drop(columns=["series", "segment"]),
    ], ignore_index=True)
    metrics = metrics[["slice", "group", "model", "n", "wape", "mape", "bias", "rmse"]]
    metrics.to_csv(results("forecast_metrics.csv"), index=False)
    print(metrics[metrics["slice"].isin(["overall", "promo"])].to_string(index=False))
    print(f"Backtest finished in {time.time() - t0:.0f}s")
    return {"rounds": rounds, "origins": origins}


if __name__ == "__main__":
    run()

"""Rolling-origin backtest: tuned LightGBM vs. three baselines (no random splits).

For each fold with origin ``T`` the models are trained on weeks ``<= T`` and forecast weeks
``T+1 .. T+H``, each week at its true horizon ``h = t - T``.

* ``lightgbm``       -- settings chosen by ``tuning.py`` (``results/best_params.json``)
* ``regression``     -- the log-log elasticity model, re-fitted on each fold's training window
* ``seasonal_naive`` -- same week last year (4-week mean if missing)
* ``moving_avg_4``   -- mean of the last 4 weeks before the origin

The LightGBM fold models also produce the promotion counterfactual used in ``promo/analysis.py``:
the same week predicted with the promotion switched off and price set back to regular.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from beverage_demand.config import processed, results, settings
from beverage_demand.elasticity.model import estimation_sample, fit_series
from beverage_demand.features.panel import load_panel
from beverage_demand.forecasting.features import build_supervised, feature_columns
from beverage_demand.forecasting.metrics import score
from beverage_demand.forecasting.models import counterfactual, load_tuned_lgb, train_lgb

MODELS = ["lightgbm", "regression", "seasonal_naive", "moving_avg_4"]


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


def load_supervised(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    fc = cfg["forecast"]
    panel = load_panel()
    sup = build_supervised(panel, cfg["scope"]["brands"], fc["horizon"], fc["lags"], fc["seasonal_lag"])
    return panel, sup[(sup["data_gap"] == 0) & sup["units"].notna()].reset_index(drop=True)


def run() -> None:
    cfg = settings()
    fc = cfg["forecast"]
    brands, H = cfg["scope"]["brands"], fc["horizon"]
    features = feature_columns(fc["lags"], fc["seasonal_lag"])
    params, rounds = load_tuned_lgb(features, fc["seed"])
    t0 = time.time()

    panel, sup = load_supervised(cfg)
    print(f"Supervised frame: {len(sup):,} rows x {len(features)} features; LightGBM rounds {rounds}")

    origins = [fc["first_origin"] + i * fc["fold_step"] for i in range(fc["n_folds"])]
    preds, cf_rows = [], []
    for i, T in enumerate(origins):
        tf = time.time()
        train = sup[sup["week"] <= T]
        model = train_lgb(train, features, params, rounds)

        test = sup[(sup["origin_week"] == T) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
        test["fold"] = i
        test["lightgbm"] = model.predict(test[features])
        test["moving_avg_4"] = test["rmean_4"]
        test["seasonal_naive"] = test["lag_52"].fillna(test["rmean_4"])
        test = test.merge(regression_forecast(panel, T, H, brands), on=["series", "store", "week"],
                          how="left")
        test["regression"] = test["regression"].fillna(test["rmean_4"])
        preds.append(test)

        # promo counterfactual: same weeks, 1 week ahead, with and without the promotion
        now = sup[(sup["h"] == 1) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
        now["fold"] = i
        now["pred_actual"] = model.predict(now[features])
        now["pred_baseline"] = model.predict(counterfactual(now)[features])
        cf_rows.append(now)
        print(f"  fold {i}: origin {T}, train rows {len(train):,}, "
              f"WAPE lightgbm {np.abs(test.units - test.lightgbm).sum() / test.units.sum():.3f} "
              f"({time.time() - tf:.0f}s)")

    pred = pd.concat(preds, ignore_index=True)
    pred[["series", "brand", "pack", "store", "week", "h", "fold", "units", "promo", "promo_type",
          "discount", *MODELS]].to_parquet(processed("backtest_predictions"), index=False)
    cf = pd.concat(cf_rows, ignore_index=True)
    cf[["series", "brand", "pack", "store", "week", "fold", "units", "promo", "promo_type",
        "discount", "unit_price", "reg_price", "unit_cost", "pred_actual", "pred_baseline"]].to_parquet(
        processed("promo_counterfactual"), index=False)
    model.save_model(str(processed("lightgbm_last_fold").with_suffix(".txt")))
    test[features].sample(min(5000, len(test)), random_state=0).to_parquet(
        processed("shap_sample"), index=False)

    pred["segment"] = np.where(pred["promo"] == 1, "promo weeks", "non-promo weeks")
    chain = pred.groupby(["series", "week"], as_index=False)[["units", *MODELS]].sum()
    metrics = pd.concat([
        score(pred, MODELS).assign(slice="overall", group="all"),
        score(pred, MODELS, by=["segment"]).rename(columns={"segment": "group"}).assign(slice="promo"),
        score(pred, MODELS, by=["series"]).rename(columns={"series": "group"}).assign(slice="series"),
        score(pred, MODELS, by=["h"]).rename(columns={"h": "group"}).assign(slice="horizon"),
        score(chain, MODELS).assign(slice="chain_level", group="all series x week"),
    ], ignore_index=True)
    metrics = metrics[["slice", "group", "model", "n", "wape", "mape", "bias", "rmse"]]
    metrics.to_csv(results("forecast_metrics.csv"), index=False)
    print(metrics[metrics["slice"].isin(["overall", "promo", "chain_level"])].to_string(index=False))
    print(f"Backtest finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()

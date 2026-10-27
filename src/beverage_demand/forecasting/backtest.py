"""Rolling-origin backtest: two ML models vs. three baselines (no random splits).

For each fold with origin ``T`` every model is trained on target weeks ``<= T`` and forecasts weeks
``T+1 .. T+H``, each week at its true horizon ``h = t - T``.

Models
------
* ``lightgbm``       -- main model, settings from ``tuning.py`` (``results/best_params.json``)
* ``hist_gbm``       -- scikit-learn HistGradientBoosting, comparison model
* ``regression``     -- the log-log elasticity model, re-fitted on each fold's training window
* ``seasonal_naive`` -- same week last year (4-week mean if missing)
* ``moving_avg_4``   -- mean of the last 4 weeks before the origin

All forecasts are written to the ``forecast_predictions`` table; accuracy is computed in SQL
(``sql/analysis/forecast_accuracy.sql``). Each run is also appended to ``experiment_runs``.
The LightGBM fold models additionally produce the promotion counterfactual (``promo_counterfactual``).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from beverage_demand.config import processed, results, settings
from beverage_demand.db import connect, query_file, write_table
from beverage_demand.elasticity.model import estimation_sample, fit_series
from beverage_demand.features.panel import load_panel
from beverage_demand.forecasting.features import build_supervised, feature_columns
from beverage_demand.forecasting.models import counterfactual, load_tuned_lgb, train_hgb, train_lgb

MODELS = ["lightgbm", "hist_gbm", "regression", "seasonal_naive", "moving_avg_4"]
BASELINES = ["regression", "seasonal_naive", "moving_avg_4"]


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


def backtest_origins(fc: dict) -> list[int]:
    return [fc["first_origin"] + i * fc["fold_step"] for i in range(fc["n_folds"])]


def accuracy_tables(con) -> pd.DataFrame:
    """All accuracy slices, computed in SQL on ``forecast_predictions``."""
    con.execute("""
        CREATE OR REPLACE VIEW forecast_chain_level AS
        SELECT series, week, model, SUM(units) AS units, SUM(forecast) AS forecast
        FROM forecast_predictions GROUP BY series, week, model""")
    slices = [("overall", "'all'", "forecast_predictions"),
              ("promo", "segment", "forecast_predictions"),
              ("series", "series", "forecast_predictions"),
              ("horizon", "h", "forecast_predictions"),
              ("series_promo", "series || ' / ' || segment", "forecast_predictions"),
              ("chain_level", "'all series x week'", "forecast_chain_level")]
    out = []
    for name, group_by, source in slices:
        t = query_file(con, "analysis/forecast_accuracy.sql", group_by=group_by, source=source)
        t = t.rename(columns={t.columns[0]: "group"}).assign(slice=name)
        out.append(t)
    m = pd.concat(out, ignore_index=True)
    m["group"] = m["group"].astype(str)
    return m[["slice", "group", "model", "n", "wape", "mape", "bias", "rmse"]]


def run() -> dict:
    cfg = settings()
    fc = cfg["forecast"]
    brands, H = cfg["scope"]["brands"], fc["horizon"]
    features = feature_columns(fc["lags"], fc["seasonal_lag"])
    params, rounds = load_tuned_lgb(features, fc["seed"])
    t0 = time.time()

    panel, sup = load_supervised(cfg)
    print(f"Supervised frame: {len(sup):,} rows x {len(features)} features; LightGBM rounds {rounds}")

    preds, cf_rows, last = [], [], None
    for i, T in enumerate(backtest_origins(fc)):
        tf = time.time()
        train = sup[sup["week"] <= T]
        lgbm = train_lgb(train, features, params, rounds)
        hgb = train_hgb(train, features, fc["seed"])

        test = sup[(sup["origin_week"] == T) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
        test["fold"] = i
        test["lightgbm"] = lgbm.predict(test[features])
        test["hist_gbm"] = hgb.predict(test[features])
        test["moving_avg_4"] = test["rmean_4"]
        test["seasonal_naive"] = test["lag_52"].fillna(test["rmean_4"])
        test = test.merge(regression_forecast(panel, T, H, brands), on=["series", "store", "week"],
                          how="left")
        test["regression"] = test["regression"].fillna(test["rmean_4"])
        preds.append(test)

        # promo counterfactual: same weeks, 1 week ahead, with and without the promotion
        now = sup[(sup["h"] == 1) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
        now["fold"] = i
        now["pred_actual"] = lgbm.predict(now[features])
        now["pred_baseline"] = lgbm.predict(counterfactual(now)[features])
        cf_rows.append(now)
        last = (lgbm, test)
        w = {m: np.abs(test.units - test[m]).sum() / test.units.sum() for m in ("lightgbm", "hist_gbm")}
        print(f"  fold {i}: origin {T}, train rows {len(train):,}, WAPE lightgbm {w['lightgbm']:.3f} "
              f"hist_gbm {w['hist_gbm']:.3f} ({time.time() - tf:.0f}s)")

    pred = pd.concat(preds, ignore_index=True)
    pred["segment"] = np.where(pred["promo"] == 1, "promo weeks", "non-promo weeks")
    long = pred.melt(id_vars=["series", "brand", "pack", "store", "week", "h", "fold", "promo",
                              "promo_type", "segment", "units"],
                     value_vars=MODELS, var_name="model", value_name="forecast")
    cf = pd.concat(cf_rows, ignore_index=True)[
        ["series", "brand", "pack", "store", "week", "fold", "units", "promo", "promo_type",
         "discount", "unit_price", "reg_price", "unit_cost", "pred_actual", "pred_baseline"]]

    lgbm, test = last
    lgbm.save_model(str(processed("lightgbm_last_fold").with_suffix(".txt")))
    with connect() as con:
        write_table(con, "forecast_predictions", long)
        write_table(con, "promo_counterfactual", cf)
        write_table(con, "shap_sample", test[features].sample(min(5000, len(test)), random_state=0))
        metrics = accuracy_tables(con)
        overall = metrics[metrics["slice"].isin(["overall", "promo"])]
        con.execute("""CREATE TABLE IF NOT EXISTS experiment_runs (
            run_at TIMESTAMP, experiment VARCHAR, model VARCHAR, params VARCHAR,
            wape_overall DOUBLE, wape_promo DOUBLE, wape_non_promo DOUBLE)""")
        for m in MODELS:
            o = overall[overall["model"] == m].set_index("group")["wape"]
            con.execute("INSERT INTO experiment_runs VALUES (now(), 'backtest', ?, ?, ?, ?, ?)",
                        [m, json.dumps({k: v for k, v in params.items() if k != "monotone_constraints"})
                         if m == "lightgbm" else "", o["all"], o["promo weeks"], o["non-promo weeks"]])
    metrics.to_csv(results("forecast_metrics.csv"), index=False)
    print(metrics[metrics["slice"].isin(["overall", "promo", "chain_level"])].to_string(index=False))
    print(f"Backtest finished in {time.time() - t0:.0f}s")
    return {"rounds": rounds}


if __name__ == "__main__":
    run()

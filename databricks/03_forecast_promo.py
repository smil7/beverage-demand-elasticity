# Databricks notebook source
# MAGIC %md
# MAGIC # 03 -- Rolling-origin forecast backtest and promotion ROI
# MAGIC
# MAGIC Same method as `src/beverage_demand/forecasting/backtest.py`: global LightGBM, direct
# MAGIC 1-8-week horizon, leakage-safe lags, seasonal-naive / moving-average / regression baselines,
# MAGIC then the promo counterfactual and ROI. Writes Delta tables `fact_forecast`,
# MAGIC `forecast_metrics`, `fact_promo_events`, `promo_roi`.
# MAGIC
# MAGIC On Databricks Free Edition use `n_folds` = 3 to keep run time short (the local run uses 9).

# COMMAND ----------

# MAGIC %pip install lightgbm==4.* statsmodels==0.14.* pyyaml

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "beverage")
dbutils.widgets.text("n_folds", "3")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
N_FOLDS = int(dbutils.widgets.get("n_folds"))

import os
import sys

REPO = os.path.abspath("..")
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pandas as pd

from beverage_demand.config import settings
from beverage_demand.forecasting.backtest import (
    MODELS, counterfactual, lgb_params, regression_forecast, train_lgb,
)
from beverage_demand.forecasting.features import build_supervised, feature_columns
from beverage_demand.forecasting.metrics import score
from beverage_demand.promo.analysis import label_events, promo_weeks, summarise

cfg = settings()
fc = cfg["forecast"]
brands, H, lags = cfg["scope"]["brands"], fc["horizon"], fc["lags"]
features = feature_columns(lags, fc["seasonal_lag"])
params = lgb_params(fc["seed"], features)


def save(pdf, name):
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{CATALOG}.{SCHEMA}.{name}"))


panel = spark.table(f"{CATALOG}.{SCHEMA}.model_panel").toPandas()
sup = build_supervised(panel, brands, H, lags, fc["seasonal_lag"])
sup = sup[(sup["data_gap"] == 0) & sup["units"].notna()].reset_index(drop=True)
print(f"supervised rows: {len(sup):,}")

# COMMAND ----------

# MAGIC %md ## Boosting rounds from the pre-backtest window, then the folds

# COMMAND ----------

origins = [fc["first_origin"] + i * fc["fold_step"] for i in range(fc["n_folds"])][-N_FOLDS:]
o0 = fc["first_origin"]
tr = sup[sup["week"] <= o0 - H]
va = sup[(sup["week"] > o0 - H) & (sup["week"] <= o0) & (sup["origin_week"] == o0 - H)]
rounds = int(train_lgb(tr, features, params, 2000, valid=va).best_iteration * 1.1)

preds, cf_rows = [], []
for i, T in enumerate(origins):
    model = train_lgb(sup[sup["week"] <= T], features, params, rounds)
    test = sup[(sup["origin_week"] == T) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
    test["fold"] = i
    test["lightgbm"] = model.predict(test[features])
    test["moving_avg_4"] = test["rmean_4"]
    test["seasonal_naive"] = test["lag_52"].fillna(test["rmean_4"])
    test = test.merge(regression_forecast(panel, T, H, brands), on=["series", "store", "week"], how="left")
    test["regression"] = test["regression"].fillna(test["rmean_4"])
    preds.append(test)
    now = sup[(sup["h"] == 1) & (sup["week"] > T) & (sup["week"] <= T + H)].copy()
    now["pred_actual"] = model.predict(now[features])
    now["pred_baseline"] = model.predict(counterfactual(now)[features])
    cf_rows.append(now)
    print(f"fold {i} origin {T}: WAPE {np.abs(test.units - test.lightgbm).sum() / test.units.sum():.3f}")

pred = pd.concat(preds, ignore_index=True)
pred["segment"] = np.where(pred["promo"] == 1, "promo weeks", "non-promo weeks")
metrics = pd.concat([score(pred, MODELS).assign(group="all"),
                     score(pred, MODELS, by=["segment"]).rename(columns={"segment": "group"})])
save(metrics, "forecast_metrics")
save(pred[["series", "store", "week", "h", "fold", "units", "promo", *MODELS]], "fact_forecast")
display(metrics.pivot_table(index="group", columns="model", values="wape").round(3).reset_index())

# COMMAND ----------

# MAGIC %md ## Promotion lift and ROI

# COMMAND ----------

cf = pd.concat(cf_rows, ignore_index=True)
p = label_events(promo_weeks(cf, cfg["promo"]["depth_buckets"]))
save(p, "fact_promo_events")
roi = summarise(p, ["brand", "pack", "promo_label", "depth_bucket"])
roi = roi[(roi["promo_weeks"] >= 50) & (roi["avg_discount"] >= 0.05)].sort_values("roi", ascending=False)
save(roi, "promo_roi")
display(summarise(p, ["pack", "depth_bucket"]))

# Databricks notebook source
# MAGIC %md
# MAGIC # 02 -- Modelling panel and price elasticities
# MAGIC
# MAGIC Builds the brand x pack x store x week panel (regular price, discount depth, promo type,
# MAGIC competitor prices) and fits the log-log elasticity model per series with store-clustered
# MAGIC standard errors. Outputs Delta tables `model_panel`, `elasticities`, `cross_elasticities`,
# MAGIC `promo_effects`.

# COMMAND ----------

# MAGIC %pip install statsmodels==0.14.* pyyaml

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "beverage")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

import os
import sys

REPO = os.path.abspath("..")
sys.path.insert(0, os.path.join(REPO, "src"))

from beverage_demand.config import settings
from beverage_demand.elasticity.model import estimation_sample
from beverage_demand.elasticity.run import collinear_pairs, cross_table, fit_all, own_table, promo_table
from beverage_demand.features.panel import build_panel

cfg = settings()
brands, packs = cfg["scope"]["brands"], cfg["scope"]["packs"]


def table(name):
    return spark.table(f"{CATALOG}.{SCHEMA}.{name}")


def save(pdf, name):
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{CATALOG}.{SCHEMA}.{name}"))

# COMMAND ----------

# filter in Spark first: only the in-scope brands/packs come back to pandas (~0.4M rows)
fact = (table("fact_sales_weekly")
        .where(f"brand IN ({','.join(repr(b) for b in brands)}) AND pack IN ({','.join(repr(p) for p in packs)})")
        .toPandas())
all_weeks = table("fact_sales_weekly").select("week").distinct().toPandas()
cal = table("dim_calendar").toPandas()
stores = table("dim_store").toPandas()

# build_panel detects chain-wide data gaps from the weeks present in `fact`; check that the
# in-scope subset covers every week the full table covers, so gaps are identical to the local run
missing = set(all_weeks["week"]) - set(fact["week"])
assert not missing, f"weeks with no in-scope sales: {sorted(missing)}"

panel = build_panel(fact, cal, stores)
save(panel.drop(columns=["week_start", "week_end"]), "model_panel")
print(f"panel: {len(panel):,} rows, {panel['series'].nunique()} series")

# COMMAND ----------

est = estimation_sample(panel)
coefs, models = fit_all(est, brands)
own = own_table(coefs, est)
cross = cross_table(coefs, est, brands, collinear_pairs(models))
save(own, "elasticities")
save(cross, "cross_elasticities")
save(promo_table(coefs), "promo_effects")
display(own[["series", "own_elasticity", "ci_low", "ci_high", "interpretation"]])

# COMMAND ----------

display(cross.pivot_table(index=["pack", "focal_brand"], columns="competitor_brand",
                          values="cross_elasticity").round(2).reset_index())

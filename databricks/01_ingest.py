# Databricks notebook source
# MAGIC %md
# MAGIC # 01 -- Ingest Dominick's soft-drink data into Delta tables
# MAGIC
# MAGIC Raw CSV / Stata files in a Unity Catalog **Volume** -> the same SQL views used locally
# MAGIC (`sql/*.sql`) -> Delta tables `dim_product`, `dim_store`, `dim_calendar`,
# MAGIC `fact_sales_weekly`, `fact_brand_week`.
# MAGIC
# MAGIC Run from a **Git folder** clone of the repository so `src/`, `sql/` and `config/` are available.
# MAGIC See `databricks/README.md` for setup.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "beverage")
dbutils.widgets.text("raw_volume", "/Volumes/workspace/beverage/raw")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
RAW = dbutils.widgets.get("raw_volume").rstrip("/")

# COMMAND ----------

import os
import sys

# notebooks run with the notebook's folder as working directory inside a Git folder
REPO = os.path.abspath("..")
sys.path.insert(0, os.path.join(REPO, "src"))

import pandas as pd

from beverage_demand.config import settings
from beverage_demand.ingest.calendar import build_dim_calendar
from beverage_demand.ingest.product import build_dim_product
from beverage_demand.ingest.spark_pipeline import MOVEMENT_SCHEMA, TABLES, run_sql_file
from beverage_demand.ingest.store import build_dim_store

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md ## Raw movement file (17.7M rows) and small dimensions

# COMMAND ----------

raw = spark.read.csv(f"{RAW}/wsdr.csv", header=True, schema=MOVEMENT_SCHEMA)
raw.createOrReplaceTempView("raw_movement")

upc = pd.read_csv(f"{RAW}/upcsdr.csv", encoding="latin-1")
demo = pd.read_stata(f"{RAW}/demo.dta")
dims = {
    "dim_product_src": build_dim_product(upc),
    "dim_store_src": build_dim_store(demo),
    "dim_calendar_src": build_dim_calendar(settings()["calendar"]["week1_start"]),
}
for name, pdf in dims.items():
    spark.createDataFrame(pdf).createOrReplaceTempView(name)

print(f"raw rows: {raw.count():,}")

# COMMAND ----------

# MAGIC %md ## Business views (identical SQL to the local pipeline)

# COMMAND ----------

from pathlib import Path

for sql_file in sorted(Path(REPO, "sql").glob("*.sql")):
    print("running", sql_file.name)
    run_sql_file(spark, sql_file)

# COMMAND ----------

for t in TABLES:
    (spark.table(t).write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{CATALOG}.{SCHEMA}.{t}"))
    print(f"{CATALOG}.{SCHEMA}.{t}: {spark.table(f'{CATALOG}.{SCHEMA}.{t}').count():,} rows")

# COMMAND ----------

# MAGIC %md ## Data-quality checks

# COMMAND ----------

from beverage_demand.ingest.quality import check_fact

fact = spark.table(f"{CATALOG}.{SCHEMA}.fact_sales_weekly").toPandas()
cal = spark.table(f"{CATALOG}.{SCHEMA}.dim_calendar").toPandas()
for c in check_fact(fact, cal):
    print(f"[{'PASS' if c.passed else ('FAIL' if c.blocking else 'WARN')}] {c.name}: {c.detail}")
assert all(c.passed or not c.blocking for c in check_fact(fact, cal))

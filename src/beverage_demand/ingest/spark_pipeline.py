"""Phase 1 ingestion pipeline in PySpark.

Raw CSV -> typed Spark DataFrame -> SQL views in ``sql/`` -> Parquet tables in ``data/processed``.
Small dimensions (product, store, calendar) are built in pandas from the codebook rules and
handed to Spark; the 17.7M-row movement file is processed entirely in Spark.

The same SQL files run unchanged on Databricks (see ``databricks/01_ingest.py``).
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import pandas as pd

from beverage_demand.config import REPO_ROOT, processed, raw_file, settings
from beverage_demand.ingest.calendar import build_dim_calendar
from beverage_demand.ingest.product import build_dim_product
from beverage_demand.ingest.store import build_dim_store

SQL_DIR = REPO_ROOT / "sql"
TABLES = ["dim_product", "dim_store", "dim_calendar", "fact_sales_weekly", "fact_brand_week"]

MOVEMENT_SCHEMA = (
    "STORE INT, UPC BIGINT, WEEK INT, MOVE BIGINT, QTY INT, PRICE DOUBLE, SALE STRING, "
    "PROFIT DOUBLE, OK INT, PRICE_HEX STRING, PROFIT_HEX STRING"
)


def get_spark(app: str = "beverage-demand"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[*]")
        .appName(app)
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def run_sql_file(spark, path: Path) -> None:
    """Execute every statement in a .sql file (statements separated by ';')."""
    text = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    )
    for stmt in (s.strip() for s in text.split(";")):
        if stmt:
            spark.sql(stmt)


def _can_use_spark_writer() -> bool:
    """Spark's Parquet writer needs Hadoop's winutils on Windows; elsewhere it just works."""
    return platform.system() != "Windows" or bool(os.environ.get("HADOOP_HOME"))


def write_table(df, name: str) -> Path:
    out = processed(name)
    if _can_use_spark_writer():
        df.write.mode("overwrite").parquet(str(out))
    else:
        # Local Windows fallback: collect through Arrow and write a single Parquet file.
        df.toPandas().to_parquet(out, index=False)
    return out


def build_dimensions() -> dict[str, pd.DataFrame]:
    upc = pd.read_csv(raw_file("upc"), encoding="latin-1")
    demo = pd.read_stata(raw_file("demographics"))
    return {
        "dim_product_src": build_dim_product(upc),
        "dim_store_src": build_dim_store(demo),
        "dim_calendar_src": build_dim_calendar(settings()["calendar"]["week1_start"]),
    }


def run() -> None:
    t0 = time.time()
    spark = get_spark()
    raw = spark.read.csv(str(raw_file("movement")), header=True, schema=MOVEMENT_SCHEMA)
    raw.createOrReplaceTempView("raw_movement")
    for name, pdf in build_dimensions().items():
        spark.createDataFrame(pdf).createOrReplaceTempView(name)

    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        run_sql_file(spark, sql_file)

    # cache the fact table: fact_brand_week is derived from it
    spark.table("fact_sales_weekly").cache()
    for t in TABLES:
        out = write_table(spark.table(t), t)
        print(f"  wrote {t:<20} -> {out.relative_to(REPO_ROOT)}")

    # row counts for the data-quality report
    counts = spark.sql(
        "SELECT (SELECT COUNT(*) FROM raw_movement) AS raw_rows, "
        "(SELECT COUNT(*) FROM movement_clean) AS clean_rows"
    ).toPandas()
    counts.to_parquet(processed("_ingest_counts"), index=False)
    print(f"Ingestion finished in {time.time() - t0:.0f}s: {counts.to_dict('records')[0]}")
    spark.stop()


if __name__ == "__main__":
    run()

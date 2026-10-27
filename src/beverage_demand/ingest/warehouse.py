"""Build the DuckDB warehouse: raw CSV -> staging -> cleaned view -> star schema.

Run:  uv run python -m beverage_demand.ingest.warehouse
"""

from __future__ import annotations

import time

import pandas as pd

from beverage_demand.config import raw_file, settings
from beverage_demand.db import connect, run_sql_file
from beverage_demand.ingest.calendar import build_dim_calendar
from beverage_demand.ingest.product import build_dim_product
from beverage_demand.ingest.store import build_dim_store

DIMENSIONS = ["dim_product", "dim_store", "dim_calendar"]


def build_dimensions() -> dict[str, pd.DataFrame]:
    upc = pd.read_csv(raw_file("upc"), encoding="latin-1")
    demo = pd.read_stata(raw_file("demographics"))
    return {
        "dim_product": build_dim_product(upc),
        "dim_store": build_dim_store(demo),
        "dim_calendar": build_dim_calendar(settings()["calendar"]["week1_start"]),
    }


def load_dimensions(con, dims: dict[str, pd.DataFrame]) -> None:
    """INSERT the pandas-built dimensions into the keyed tables from 01_schema.sql."""
    run_sql_file(con, "warehouse/01_schema.sql")
    for name, df in dims.items():
        cols = [r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()]
        con.register("_dim", df[cols])
        con.execute(f"INSERT INTO {name} SELECT * FROM _dim")
        con.unregister("_dim")


def run() -> None:
    t0 = time.time()
    with connect() as con:
        load_dimensions(con, build_dimensions())
        movement_csv = raw_file("movement").as_posix()
        run_sql_file(con, "warehouse/02_staging.sql", movement_csv=movement_csv)
        run_sql_file(con, "warehouse/03_movement_clean.sql")
        run_sql_file(con, "warehouse/04_fact_sales_weekly.sql")
        run_sql_file(con, "warehouse/05_fact_brand_week.sql")
        # keep a log of each load so row counts are queryable later
        con.execute("""
            CREATE TABLE IF NOT EXISTS etl_run_log (
                loaded_at TIMESTAMP, raw_rows BIGINT, clean_rows BIGINT, fact_rows BIGINT)""")
        con.execute("""
            INSERT INTO etl_run_log
            SELECT now(), (SELECT COUNT(*) FROM stg_movement), (SELECT COUNT(*) FROM movement_clean),
                   (SELECT COUNT(*) FROM fact_sales_weekly)""")
        for t in [*DIMENSIONS, "stg_movement", "fact_sales_weekly", "fact_brand_week"]:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<20} {n:>12,} rows")
    print(f"Warehouse built in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()

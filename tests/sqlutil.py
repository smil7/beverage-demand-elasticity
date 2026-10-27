"""Helpers for testing the SQL in an in-memory DuckDB database."""

from __future__ import annotations

import pandas as pd

from beverage_demand.db import connect, run_sql_file, sql_list
from beverage_demand.ingest.calendar import build_dim_calendar
from beverage_demand.ingest.warehouse import load_dimensions

FACT_COLS = ["store", "brand", "pack", "week", "units", "dollars", "cogs", "eq_cases", "unit_price",
             "units_sale_b", "units_sale_s", "units_sale_c", "units_sale_g",
             "any_sale_b", "any_sale_s", "any_sale_c", "any_sale_g", "n_upcs"]


def fact_rows(store, brand, pack, weeks, prices, units=None, codes=None, margin=0.25):
    """Rows for fact_sales_weekly; ``codes`` holds a SALE code ('B', 'S', ...) or '' per week."""
    n = len(weeks)
    units = units or [10] * n
    codes = codes or [""] * n
    rows = []
    for w, p, u, c in zip(weeks, prices, units, codes, strict=True):
        r = {"store": store, "brand": brand, "pack": pack, "week": w, "units": u,
             "dollars": p * u, "cogs": p * u * (1 - margin), "eq_cases": u, "unit_price": p,
             "n_upcs": 1}
        for k in "bscg":
            r[f"units_sale_{k}"] = u if c.lower() == k else 0
            r[f"any_sale_{k}"] = int(c.lower() == k)
        rows.append(r)
    return rows


def db_with_fact(rows: list[dict]):
    """In-memory warehouse with the real schema, a calendar, two stores and the given fact rows."""
    con = connect(":memory:")
    stores = pd.DataFrame({
        "store": [1, 2], "city": ["A", "B"], "zone": [1, 2], "price_tier": ["High", "Low"],
        "lat": [0.0, 0.0], "lon": [0.0, 0.0], "income_log_median": [10.0, 10.5],
        "pct_college": [0.2, 0.3], "pct_age60plus": [0.1, 0.2], "pct_ethnic": [0.1, 0.1],
        "avg_household_size": [2.5, 2.7], "pop_density": [0.001, 0.002]})
    product = pd.DataFrame({"upc": [1], "description": ["X"], "size": ["2 LT"], "case_pack": [8],
                            "com_code": [1], "brand": ["Coke"], "pack": ["2L"], "items_per_pack": [1],
                            "oz_per_item": [67.6], "oz_per_unit": [67.6]})
    load_dimensions(con, {"dim_product": product, "dim_store": stores,
                          "dim_calendar": build_dim_calendar(n_weeks=60)})
    # real DDL from the warehouse scripts, on an empty staging table
    con.execute("""CREATE TABLE stg_movement (STORE INTEGER, UPC BIGINT, WEEK INTEGER, MOVE BIGINT,
        QTY INTEGER, PRICE DOUBLE, SALE VARCHAR, PROFIT DOUBLE, OK INTEGER)""")
    run_sql_file(con, "warehouse/03_movement_clean.sql")
    run_sql_file(con, "warehouse/04_fact_sales_weekly.sql")
    if rows:
        df = pd.DataFrame(rows)[FACT_COLS]
        con.register("_f", df)
        con.execute("INSERT INTO fact_sales_weekly SELECT * FROM _f")
        con.unregister("_f")
    return con


def build_panel_sql(con, brands, packs, min_weeks=1, reg_window=13, threshold=0.05):
    run_sql_file(con, "features/model_panel.sql", brands=sql_list(brands), packs=sql_list(packs),
                 reg_preceding=reg_window - 1, threshold=threshold, min_weeks=min_weeks)
    return con.execute("SELECT * FROM model_panel ORDER BY series, store, week").df()

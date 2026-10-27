"""Modelling panel: built in SQL (``sql/features/model_panel.sql``) inside the warehouse.

One row per in-scope brand x pack x store x week with the features every model uses:

* ``reg_price`` -- regular (shelf) price: highest price in the trailing 13 weeks incl. this week
* ``discount``  -- ``1 - price / reg_price``
* ``promo_type`` -- SALE code (B, S, C, G) or ``U`` = unflagged price cut >= 5%, else ``none``
* ``lp_<brand>`` -- log price of each in-scope brand in the same pack (competitor prices)
"""

from __future__ import annotations

import duckdb
import pandas as pd

from beverage_demand.config import settings
from beverage_demand.db import connect, run_sql_file, sql_list

PROMO_TYPES = ["B", "S", "C", "G", "U"]
PROMO_LABELS = {
    "B": "Bonus buy (B)",
    "S": "Simple price reduction (S)",
    "C": "Coupon (C)",
    "G": "Undocumented code (G)",
    "U": "Unflagged price cut (U)",
}


def series_name(brand: str, pack: str) -> str:
    return f"{brand} | {pack}"


def brand_col(brand: str) -> str:
    """Column holding a brand's log price, e.g. ``Diet Coke`` -> ``lp_diet_coke``."""
    return "lp_" + brand.replace(" ", "_").lower()


def build_panel(con: duckdb.DuckDBPyConnection) -> None:
    cfg = settings()
    run_sql_file(
        con, "features/model_panel.sql",
        brands=sql_list(cfg["scope"]["brands"]),
        packs=sql_list(cfg["scope"]["packs"]),
        reg_preceding=cfg["promo"]["regular_price_window"] - 1,
        threshold=cfg["promo"]["discount_threshold"],
        min_weeks=cfg["scope"]["min_weeks_per_store_series"],
    )


def run() -> None:
    with connect() as con:
        build_panel(con)
        n, s, ss = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT series), COUNT(DISTINCT (series, store)) FROM model_panel"
        ).fetchone()
    print(f"model_panel: {n:,} rows, {s} series, {ss} store-series")


def load_panel(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM model_panel ORDER BY series, store, week"
    if con is not None:
        return con.execute(sql).df()
    with connect(read_only=True) as c:
        return c.execute(sql).df()


if __name__ == "__main__":
    run()

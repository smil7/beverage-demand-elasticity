"""Load the raw Dominick's soft-drink files, clean them and aggregate to weekly sales.

Run:  uv run python -m beverage_demand.data.load

Output (data/processed/, git-ignored):
    sales_weekly.parquet  brand x pack x store x week
    brand_weekly.parquet  brand x week, chain level (8-oz equivalent cases)
    products / stores / calendar .parquet
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from beverage_demand.config import processed, raw_file, settings
from beverage_demand.data.calendar import build_dim_calendar
from beverage_demand.data.products import build_dim_product
from beverage_demand.data.stores import build_dim_store

MOVEMENT_DTYPES = {"STORE": "int16", "UPC": "int64", "WEEK": "int16", "MOVE": "int32",
                   "QTY": "int16", "PRICE": "float64", "SALE": "string", "PROFIT": "float64",
                   "OK": "int8"}
SALE_CODES = ["B", "S", "C", "G"]  # bonus buy, simple price reduction, coupon, undocumented


def read_movement() -> pd.DataFrame:
    return pd.read_csv(raw_file("movement"), usecols=list(MOVEMENT_DTYPES), dtype=MOVEMENT_DTYPES)


def clean_movement(m: pd.DataFrame) -> pd.DataFrame:
    """Codebook rules: drop suspect rows (OK = 0) and weeks the item was not on sale (PRICE <= 0).

    PRICE is the price of a bundle of QTY items and MOVE counts items, so
    unit price = PRICE / QTY and dollar sales = PRICE * MOVE / QTY.
    PROFIT is the retailer's gross margin in %, so cost = dollars * (1 - PROFIT / 100).
    """
    m = m[(m["OK"] == 1) & (m["PRICE"] > 0) & (m["MOVE"] >= 0) & (m["QTY"] > 0)]
    out = pd.DataFrame({
        "store": m["STORE"], "upc": m["UPC"], "week": m["WEEK"], "units": m["MOVE"],
        "unit_price": m["PRICE"] / m["QTY"],
        "dollars": m["PRICE"] * m["MOVE"] / m["QTY"],
        "sale_code": m["SALE"].str.strip().replace("", pd.NA),
    })
    # margins outside [-100, 100] are recording errors -> cost unknown
    ok_margin = m["PROFIT"].between(-100, 100)
    out["cogs"] = (out["dollars"] * (1 - m["PROFIT"] / 100)).where(ok_margin)
    return out


def aggregate_weekly(clean: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Brand x pack x store x week. Price = volume-weighted average (dollars / units)."""
    d = clean.merge(products[["upc", "brand", "pack", "oz_per_unit"]], on="upc", how="inner")
    d["eq_cases"] = d["units"] * d["oz_per_unit"] / 192.0  # 8-oz equivalent cases
    for c in SALE_CODES:
        is_c = d["sale_code"] == c
        d[f"units_sale_{c.lower()}"] = d["units"].where(is_c.fillna(False), 0)
        d[f"any_sale_{c.lower()}"] = is_c.fillna(False).astype(int)
    keys = ["store", "brand", "pack", "week"]
    agg = {"units": "sum", "dollars": "sum", "cogs": lambda s: s.sum(min_count=1),
           "eq_cases": "sum", "unit_price": "mean", "upc": "nunique",
           **{f"units_sale_{c.lower()}": "sum" for c in SALE_CODES},
           **{f"any_sale_{c.lower()}": "max" for c in SALE_CODES}}
    f = d.groupby(keys, as_index=False).agg(agg).rename(columns={"upc": "n_upcs"})
    # weeks with zero units keep the simple average shelf price
    f["unit_price"] = np.where(f["units"] > 0, f["dollars"] / f["units"].where(f["units"] > 0),
                               f["unit_price"])
    return f


def brand_weekly(f: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    b = f.groupby(["brand", "week"], as_index=False).agg(
        eq_cases=("eq_cases", "sum"), units=("units", "sum"), dollars=("dollars", "sum"),
        cogs=("cogs", "sum"), n_stores=("store", "nunique"))
    b["gross_margin"] = b["dollars"] - b["cogs"]
    return b.merge(cal[["week", "week_start"]], on="week")


def check(f: pd.DataFrame, raw_rows: int, clean_rows: int) -> None:
    """A few sanity checks; stop if the basics are wrong."""
    assert not f.duplicated(["store", "brand", "pack", "week"]).any(), "duplicate keys"
    assert (f["units"] >= 0).all(), "negative volumes"
    assert (f["unit_price"] > 0).all(), "non-positive prices"
    weeks = set(range(f["week"].min(), f["week"].max() + 1))
    missing = sorted(weeks - set(f["week"]))
    margin = 1 - f["cogs"] / f["dollars"].where(f["dollars"] > 0)
    odd = ((margin < -0.5) | (margin > 0.9)).mean()
    print(f"  raw rows {raw_rows:,} -> clean {clean_rows:,} -> weekly {len(f):,}")
    print(f"  stores {f['store'].nunique()}, weeks {f['week'].nunique()}, "
          f"weeks with no data in any store: {missing}")
    print(f"  rows with implausible margin (kept, cost ignored later): {odd:.2%}")


def run() -> None:
    t0 = time.time()
    raw = read_movement()
    clean = clean_movement(raw)
    products = build_dim_product(pd.read_csv(raw_file("upc"), encoding="latin-1"))
    stores = build_dim_store(pd.read_stata(raw_file("demographics")))
    cal = build_dim_calendar(settings()["calendar"]["week1_start"])

    sales = aggregate_weekly(clean, products)
    check(sales, len(raw), len(clean))
    sales.to_parquet(processed("sales_weekly"), index=False)
    brand_weekly(sales, cal).to_parquet(processed("brand_weekly"), index=False)
    products.to_parquet(processed("products"), index=False)
    stores.to_parquet(processed("stores"), index=False)
    cal.to_parquet(processed("calendar"), index=False)
    pd.Series({"raw_rows": len(raw), "clean_rows": len(clean)}).to_json(
        processed("row_counts").with_suffix(".json"))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()

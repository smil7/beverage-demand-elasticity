"""Modelling panel: one row per in-scope brand x pack x store x week with price/promo features.

Definitions used everywhere downstream:

* ``reg_price`` -- regular (shelf) price: the highest unit price in the trailing
  ``regular_price_window`` weeks *including* the current week. Prices are set by the retailer in
  advance, so the current-week price is known when forecasting (see docs/methodology.md).
* ``discount`` -- depth of the temporary price reduction, ``1 - price / reg_price``.
* ``promo`` -- 1 if any SALE code is set or the discount is >= the configured threshold. The codebook
  warns that SALE codes are recorded inconsistently, hence the price-based rule.
* ``promo_type`` -- dominant SALE code by units (B = bonus buy, S = simple price reduction,
  C = coupon, G = undocumented code) or ``U`` (unflagged price cut) or ``none``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from beverage_demand.config import processed, settings

PROMO_TYPES = ["B", "S", "C", "G", "U"]
PROMO_LABELS = {
    "B": "Bonus buy (B)",
    "S": "Simple price reduction (S)",
    "C": "Coupon (C)",
    "G": "Undocumented code (G)",
    "U": "Unflagged price cut (U)",
}
SERIES_KEYS = ["store", "brand", "pack"]


def series_name(brand: str, pack: str) -> str:
    return f"{brand} | {pack}"


def brand_col(brand: str) -> str:
    """Column name holding a brand's log price, e.g. ``Diet Coke`` -> ``lp_diet_coke``."""
    return "lp_" + brand.replace(" ", "_").lower()


def complete_grid(f: pd.DataFrame, chain_weeks: set[int]) -> pd.DataFrame:
    """Reindex every store x series to consecutive weeks between its first and last sale.

    Weeks where the chain has no data at all are kept with ``data_gap = 1`` and NaN units so lags
    respect calendar time; other missing weeks mean "not sold that week" -> 0 units.
    """
    parts = []
    for key, g in f.groupby(SERIES_KEYS, sort=False):
        weeks = np.arange(g["week"].min(), g["week"].max() + 1)
        g = g.drop(columns=SERIES_KEYS).set_index("week").reindex(weeks)
        g.index.name = "week"
        g = g.reset_index()
        g["store"], g["brand"], g["pack"] = key
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["data_gap"] = (~out["week"].isin(chain_weeks)).astype(int)
    out["stocked"] = out["units"].notna().astype(int)
    fill0 = [c for c in out.columns if c.startswith(("units", "any_sale", "dollars", "cogs", "eq_"))]
    nongap = out["data_gap"] == 0
    out.loc[nongap, fill0] = out.loc[nongap, fill0].fillna(0)
    return out


def add_price_promo(df: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    """Regular price, discount depth, promo flag/type and post-promotion flags (per store x series)."""
    df = df.sort_values([*SERIES_KEYS, "week"]).copy()
    # carry the last known shelf price through weeks the item was not sold
    df["unit_price"] = df.groupby(SERIES_KEYS, sort=False)["unit_price"].ffill()
    df["unit_price"] = df.groupby(SERIES_KEYS, sort=False)["unit_price"].bfill()
    df["reg_price"] = df.groupby(SERIES_KEYS, sort=False)["unit_price"].transform(
        lambda s: s.rolling(window, min_periods=1).max()
    )
    df["discount"] = (1 - df["unit_price"] / df["reg_price"]).clip(0, 0.9)

    sale_units = df[["units_sale_b", "units_sale_s", "units_sale_c", "units_sale_g"]].to_numpy(float)
    sale_any = df[["any_sale_b", "any_sale_s", "any_sale_c", "any_sale_g"]].to_numpy(float)
    # dominant code by units; fall back to "any flag set" when units are zero
    score = np.nan_to_num(sale_units) + 1e-6 * np.nan_to_num(sale_any)
    has_code = np.nan_to_num(sale_any).max(axis=1) > 0
    code = np.array(["B", "S", "C", "G"])[score.argmax(axis=1)]
    price_cut = df["discount"].to_numpy() >= threshold
    df["promo_type"] = np.where(has_code, code, np.where(price_cut, "U", "none"))
    df["promo"] = (df["promo_type"] != "none").astype(int)
    for t in PROMO_TYPES:
        df[f"promo_{t.lower()}"] = (df["promo_type"] == t).astype(int)

    g = df.groupby(SERIES_KEYS, sort=False)["promo"]
    lag1, lag2 = g.shift(1).fillna(0), g.shift(2).fillna(0)
    df["post_promo_1"] = ((lag1 == 1) & (df["promo"] == 0)).astype(int)
    df["post_promo_2"] = ((lag2 == 1) & (lag1 == 0) & (df["promo"] == 0)).astype(int)

    # unit cost from the retailer margin; implausible margins -> carry the last valid cost
    margin = 1 - df["cogs"] / df["dollars"].where(df["dollars"] > 0)
    unit_cost = (df["cogs"] / df["units"].where(df["units"] > 0)).where(margin.between(-0.5, 0.9))
    df["unit_cost"] = unit_cost
    df["unit_cost"] = df.groupby(SERIES_KEYS, sort=False)["unit_cost"].ffill()
    df["unit_cost"] = df.groupby(SERIES_KEYS, sort=False)["unit_cost"].bfill()
    return df


def add_competitor_prices(df: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
    """Log price of every in-scope brand in the same pack, store and week (``lp_<brand>``).

    Also ``comp_discount``: average discount depth of the *other* in-scope brands in that pack.
    """
    key = ["store", "pack", "week"]
    wide_p = df.pivot_table(index=key, columns="brand", values="unit_price", aggfunc="first")
    wide_p = wide_p.reindex(columns=brands)
    # brands not sold that week: last known price in that store, then the pack-level median
    wide_p = wide_p.groupby(level=["store", "pack"]).ffill()
    wide_p = wide_p.groupby(level=["store", "pack"]).bfill()
    wide_p = wide_p.fillna(wide_p.groupby(level="pack").transform("median"))
    lp = np.log(wide_p)
    lp.columns = [brand_col(b) for b in lp.columns]
    df = df.merge(lp.reset_index(), on=key, how="left")

    wide_d = df.pivot_table(index=key, columns="brand", values="discount", aggfunc="first")
    wide_d = wide_d.reindex(columns=brands)
    tot = wide_d.sum(axis=1).rename("_tot")
    cnt = wide_d.notna().sum(axis=1).rename("_cnt")
    df = df.merge(pd.concat([tot, cnt], axis=1).reset_index(), on=key, how="left")
    others = (df["_cnt"] - 1).clip(lower=1)
    df["comp_discount"] = ((df["_tot"] - df["discount"].fillna(0)) / others).fillna(0)
    return df.drop(columns=["_tot", "_cnt"])


def build_panel(fact: pd.DataFrame, cal: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    cfg = settings()
    brands, packs = cfg["scope"]["brands"], cfg["scope"]["packs"]
    f = fact[fact["brand"].isin(brands) & fact["pack"].isin(packs)]
    chain_weeks = set(fact["week"].unique())
    df = complete_grid(f, chain_weeks)
    df = add_price_promo(df, cfg["promo"]["regular_price_window"], cfg["promo"]["discount_threshold"])
    df = add_competitor_prices(df, brands)

    # keep store x series with enough selling weeks to estimate anything
    sells = df[df["units"] > 0].groupby(SERIES_KEYS).size().rename("n_sell")
    df = df.merge(sells.reset_index(), on=SERIES_KEYS, how="left")
    df = df[df["n_sell"] >= cfg["scope"]["min_weeks_per_store_series"]].drop(columns="n_sell")

    df["series"] = [series_name(b, p) for b, p in zip(df["brand"], df["pack"], strict=True)]
    df["ln_price"] = np.log(df["unit_price"])
    df["ln_reg_price"] = np.log(df["reg_price"])
    df = df.merge(cal, on="week", how="left").merge(stores, on="store", how="left")
    df["price_tier"] = df["price_tier"].fillna("Unknown")
    return df.sort_values(["series", "store", "week"]).reset_index(drop=True)


def run() -> pd.DataFrame:
    fact = pd.read_parquet(processed("fact_sales_weekly"))
    cal = pd.read_parquet(processed("dim_calendar"))
    stores = pd.read_parquet(processed("dim_store"))
    panel = build_panel(fact, cal, stores)
    panel.to_parquet(processed("model_panel"), index=False)
    print(
        f"model_panel: {len(panel):,} rows, {panel['series'].nunique()} series, "
        f"{panel.groupby(['series', 'store']).ngroups} store-series"
    )
    return panel


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(processed("model_panel"))


if __name__ == "__main__":
    run()

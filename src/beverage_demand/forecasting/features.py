"""Leakage-safe supervised features for direct multi-horizon forecasting.

A forecast is made at an *origin* week ``T`` (the last week with known sales) for a target week
``t = T + h``, ``h = 1..H``. Every sales-based feature therefore uses data up to ``t - h`` only:

* ``lag_k``   = units at ``t - (h + k - 1)``  -> at h = 1, lag_1 is last week's sales
* ``lag_52``  = units at ``t - 52`` (same week last year; valid because H <= 52)
* ``rmean_w`` / ``rstd_w`` = mean / std of units over the ``w`` weeks ending at ``t - h``

Price and promotion features describe the *target* week. They are treated as known in advance
because the retailer plans prices and promotions weeks ahead (see docs/methodology.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from beverage_demand.elasticity.model import HOLIDAY_COLS
from beverage_demand.features.panel import PROMO_TYPES, brand_col

GROUP = ["series", "store"]
KNOWN_IN_ADVANCE = [
    "ln_price", "ln_reg_price", "discount", "price_vs_comp", "comp_discount",
    *[f"promo_{t.lower()}" for t in PROMO_TYPES], "post_promo_1", "post_promo_2",
]
CALENDAR = [*HOLIDAY_COLS, "week_of_year", "sin1", "cos1", "sin2", "cos2"]
STORE_ATTRS = ["income_log_median", "pct_college", "pct_age60plus", "pct_ethnic",
               "avg_household_size", "pop_density"]
CATEGORICAL = ["series_code", "tier_code"]


def add_static(panel: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
    """Features that do not depend on the horizon."""
    df = panel.sort_values([*GROUP, "week"]).reset_index(drop=True).copy()
    lp = df[[brand_col(b) for b in brands]]
    df["price_vs_comp"] = df["ln_price"] - lp.mean(axis=1)
    df["series_code"] = df["series"].astype("category").cat.codes
    df["tier_code"] = df["price_tier"].astype("category").cat.codes
    # units are NaN in chain-wide data gaps -> lags/rolling stats ignore them
    g = df.groupby(GROUP, sort=False)["units"]
    for w in (4, 13):
        df[f"_rmean_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 2)).mean())
        df[f"_rstd_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 2)).std())
    return df


def horizon_frame(df: pd.DataFrame, h: int, lags: list[int], seasonal_lag: int = 52) -> pd.DataFrame:
    """Rows of ``df`` re-expressed as forecasts made ``h`` weeks before their target week."""
    g = df.groupby(GROUP, sort=False)
    out = pd.DataFrame(index=df.index)
    out["h"] = h
    out["origin_week"] = df["week"] - h
    for k in lags:
        out[f"lag_{k}"] = g["units"].shift(h + k - 1)
    out[f"lag_{seasonal_lag}"] = g["units"].shift(seasonal_lag)
    for w in (4, 13):
        out[f"rmean_{w}"] = g[f"_rmean_{w}"].shift(h)
        out[f"rstd_{w}"] = g[f"_rstd_{w}"].shift(h)
    return out


def feature_columns(lags: list[int], seasonal_lag: int = 52) -> list[str]:
    sales = [f"lag_{k}" for k in lags] + [f"lag_{seasonal_lag}", "rmean_4", "rmean_13",
                                           "rstd_4", "rstd_13", "h"]
    return [*sales, *KNOWN_IN_ADVANCE, *CALENDAR, *STORE_ATTRS, *CATEGORICAL]


def build_supervised(panel: pd.DataFrame, brands: list[str], horizon: int, lags: list[int],
                     seasonal_lag: int = 52) -> pd.DataFrame:
    """Stack one copy of the panel per horizon 1..H (direct multi-horizon strategy)."""
    df = add_static(panel, brands)
    keep = ["series", "brand", "pack", "store", "week", "units", "data_gap", "promo", "promo_type",
            "unit_price", "reg_price", "unit_cost", *KNOWN_IN_ADVANCE, *CALENDAR, *STORE_ATTRS,
            *CATEGORICAL, "price_tier", "year",
            *[brand_col(b) for b in brands]]
    keep = list(dict.fromkeys(keep))
    base = df[keep]
    frames = [pd.concat([base, horizon_frame(df, h, lags, seasonal_lag)], axis=1)
              for h in range(1, horizon + 1)]
    sup = pd.concat(frames, ignore_index=True)
    float_cols = sup.select_dtypes("float64").columns
    sup[float_cols] = sup[float_cols].astype(np.float32)
    return sup

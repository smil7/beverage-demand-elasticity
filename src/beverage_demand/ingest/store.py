"""Store dimension from the Dominick's store-demographics file (``demo.dta``)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_dim_store(demo: pd.DataFrame) -> pd.DataFrame:
    """One row per store: zone, price tier and a handful of trading-area demographics.

    Price tier comes from the ``priclow`` / ``pricmed`` / ``prichigh`` dummies (codebook Part 6).
    Stores with none of them set (opened after the zone data was collected) are ``Unknown``.
    """
    d = demo.dropna(subset=["store"]).copy()
    d["store"] = d["store"].astype(int)
    tier = np.select(
        [d["priclow"] == 1, d["pricmed"] == 1, d["prichigh"] == 1],
        ["Low", "Medium", "High"],
        default="Unknown",
    )
    out = pd.DataFrame(
        {
            "store": d["store"],
            "city": d["city"].astype(str).str.strip().str.title(),
            "zone": d["zone"].astype("Int64"),
            "price_tier": tier,
            "lat": d["lat"] / 1e4 if d["lat"].abs().max() > 1000 else d["lat"],
            "lon": d["long"] / 1e4 if d["long"].abs().max() > 1000 else d["long"],
            "income_log_median": d["income"],
            "pct_college": d["educ"],
            "pct_age60plus": d["age60"],
            "pct_ethnic": d["ethnic"],
            "avg_household_size": d["hsizeavg"],
            "pop_density": d["density"],
        }
    )
    return out.drop_duplicates("store").sort_values("store").reset_index(drop=True)

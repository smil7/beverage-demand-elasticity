"""Synthetic store x week demand panel with *known* elasticities, used by the test-suite and CI.

Demand is simulated from the same log-log structure the project estimates::

    ln(units) = a_store + OWN * ln(price) + CROSS * ln(competitor price)
                + PROMO_LIFT * promo_s + seasonality + noise

so a correct estimator must recover OWN and CROSS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from beverage_demand.data.calendar import build_dim_calendar
from beverage_demand.features.panel import PROMO_TYPES, brand_col

TRUE_OWN = -2.0
TRUE_CROSS = 0.5
TRUE_PROMO_LOG_LIFT = 0.3
BRANDS = ["Coke", "Pepsi"]


def make_panel(n_stores: int = 12, n_weeks: int = 160, seed: int = 0) -> pd.DataFrame:
    """Panel for focal brand ``Coke`` (pack 2L) with Pepsi as the competitor."""
    rng = np.random.default_rng(seed)
    cal = build_dim_calendar(n_weeks=n_weeks)
    rows = []
    for store in range(1, n_stores + 1):
        base = rng.normal(4.0, 0.3)
        reg = rng.uniform(1.3, 1.6)
        for _, c in cal.iterrows():
            promo = rng.random() < 0.3
            depth = rng.uniform(0.1, 0.3) if promo else 0.0
            price = reg * (1 - depth) * np.exp(rng.normal(0, 0.03))
            comp = reg * np.exp(rng.normal(0, 0.12))
            ln_units = (
                base + TRUE_OWN * np.log(price) + TRUE_CROSS * np.log(comp)
                + TRUE_PROMO_LOG_LIFT * promo + 0.2 * c["sin1"] + rng.normal(0, 0.15)
            )
            rows.append({"store": store, "week": int(c["week"]), "unit_price": price,
                         "reg_price": reg, "comp_price": comp, "promo": int(promo),
                         "units": float(np.round(np.exp(ln_units)))})
    df = pd.DataFrame(rows).merge(cal, on="week")
    df["brand"], df["pack"], df["series"] = "Coke", "2L", "Coke | 2L"
    df["ln_price"] = np.log(df["unit_price"])
    df[brand_col("Pepsi")] = np.log(df["comp_price"])
    df["discount"] = 1 - df["unit_price"] / df["reg_price"]
    df["promo_type"] = np.where(df["promo"] == 1, "S", "none")
    for t in PROMO_TYPES:
        df[f"promo_{t.lower()}"] = (df["promo_type"] == t).astype(int)
    df["post_promo_1"] = 0
    df["post_promo_2"] = 0
    df["data_gap"] = 0
    df["price_tier"] = np.where(df["store"] % 2 == 0, "High", "Low")
    return df

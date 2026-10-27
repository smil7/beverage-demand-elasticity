"""Phase 6: what-if price/promotion scenarios from the Phase 3 elasticities.

Assumptions (state them whenever results are shown):

* constant elasticities: volume multiplier = (1 + price change) ** elasticity
* competitors do not react (their prices stay at regular price)
* effects on competitors come only through identified cross-price elasticities (same pack)
* baseline = average non-promoted weekly volume over the last ``BASE_WEEKS`` weeks of data
* uncertainty: coefficients drawn independently from N(estimate, SE^2) -> 5th-95th percentiles
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from beverage_demand.config import results, settings
from beverage_demand.features.panel import load_panel, series_name

BASE_WEEKS = 26


@dataclass(frozen=True)
class Baseline:
    units: float        # weekly units across the selected stores, regular price, no promo
    price: float        # regular unit price
    cost: float         # unit cost


@lru_cache(maxsize=1)
def _panel_tail() -> pd.DataFrame:
    p = load_panel()
    last = p["week"].max()
    return p[(p["week"] > last - BASE_WEEKS) & (p["data_gap"] == 0)]


@lru_cache(maxsize=1)
def _coefs() -> pd.DataFrame:
    return pd.read_csv(results("elasticity_coefficients.csv"))


@lru_cache(maxsize=1)
def _cross() -> pd.DataFrame:
    return pd.read_csv(results("cross_elasticities.csv"))


def baseline(series: str, stores: tuple[int, ...] | None = None) -> Baseline:
    d = _panel_tail()
    d = d[(d["series"] == series) & (d["promo"] == 0)]
    if stores:
        d = d[d["store"].isin(stores)]
    if d.empty:
        raise ValueError(f"No non-promoted weeks for {series} in the selected stores")
    weekly = d.groupby("store")["units"].mean().sum()
    return Baseline(units=float(weekly), price=float(d["reg_price"].mean()),
                    cost=float(d["unit_cost"].median()))


def coef(series: str, term: str) -> tuple[float, float]:
    c = _coefs()
    row = c[(c["series"] == series) & (c["term"] == term)]
    if row.empty:
        return 0.0, 0.0
    return float(row["coef"].iloc[0]), float(row["se"].iloc[0])


def _summarise(draws: np.ndarray) -> tuple[float, float, float]:
    return float(np.median(draws)), float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def simulate(brand: str, pack: str, price_change_pct: float, promo_type: str | None = None,
             weeks: int = 4, stores: tuple[int, ...] | None = None,
             n_sims: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Volume, revenue and gross margin over ``weeks`` for the target series and its competitors."""
    n_sims = n_sims or settings()["scenario"]["n_sims"]
    rng = np.random.default_rng(seed)
    target = series_name(brand, pack)
    change = np.log1p(price_change_pct / 100)
    rows = []

    # target series
    b = baseline(target, stores)
    own, own_se = coef(target, "ln_price")
    log_mult = rng.normal(own, own_se, n_sims) * change
    if promo_type:
        pc, pse = coef(target, f"promo_{promo_type.lower()}")
        log_mult = log_mult + rng.normal(pc, pse, n_sims)
    mult = np.exp(log_mult)
    new_price = b.price * (1 + price_change_pct / 100)
    rows.append(_row(target, "target", b, mult, new_price, weeks))

    # competitors in the same pack that respond to this brand's price
    cross = _cross()
    resp = cross[(cross["pack"] == pack) & (cross["competitor_brand"] == brand) & cross["identified"]]
    for _, r in resp.iterrows():
        cb = baseline(r["series"], stores)
        se = (r["ci_high"] - r["ci_low"]) / (2 * 1.96)
        cmult = np.exp(rng.normal(r["cross_elasticity"], se, n_sims) * change)
        rows.append(_row(r["series"], "competitor", cb, cmult, cb.price, weeks))
    return pd.DataFrame(rows)


def _row(series: str, role: str, b: Baseline, mult: np.ndarray, price: float, weeks: int) -> dict:
    base_units = b.units * weeks
    units = base_units * mult
    base_rev, base_gm = base_units * b.price, base_units * (b.price - b.cost)
    rev, gm = units * price, units * (price - b.cost)
    u, u_lo, u_hi = _summarise(units)
    g, g_lo, g_hi = _summarise(gm)
    r, _, _ = _summarise(rev)
    return {
        "series": series, "role": role, "price_base": b.price, "price_scenario": price,
        "units_base": base_units, "units_scenario": u, "units_p5": u_lo, "units_p95": u_hi,
        "units_change_pct": u / base_units - 1,
        "revenue_base": base_rev, "revenue_scenario": r, "revenue_change_pct": r / base_rev - 1,
        "gm_base": base_gm, "gm_scenario": g, "gm_p5": g_lo, "gm_p95": g_hi,
        "gm_change_pct": g / base_gm - 1,
    }


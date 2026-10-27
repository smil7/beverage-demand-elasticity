"""Log-log demand model (Phase 3).

For one brand x pack series, at store x week level::

    ln(units) = b_own * ln(own price) + sum_j b_j * ln(price of brand j, same pack)
              + promo-type dummies + post-promotion dummies + holiday dummies
              + annual Fourier terms + year (or trend) effects + store fixed effects + e

``b_own`` is the own-price elasticity, ``b_j`` the cross-price elasticities. Standard errors are
clustered by store because the errors of one store are correlated over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from beverage_demand.features.panel import PROMO_TYPES, brand_col

HOLIDAY_COLS = [
    "hol_new_year", "hol_super_bowl", "hol_presidents_day", "hol_easter", "hol_memorial_day",
    "hol_july_4", "hol_labor_day", "hol_halloween", "hol_thanksgiving", "hol_christmas",
]
FOURIER_COLS = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3"]
PROMO_COLS = [f"promo_{t.lower()}" for t in PROMO_TYPES]
POST_COLS = ["post_promo_1", "post_promo_2"]


def estimation_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable in a log model: positive units, no chain-wide data gap, no extreme prices."""
    d = df[(df["units"] > 0) & (df["data_gap"] == 0)].copy()
    med = d.groupby("series")["unit_price"].transform("median")
    return d[(d["unit_price"] < 3 * med) & (d["unit_price"] > med / 3)]


@dataclass
class DesignSpec:
    """Column layout of a fitted model so new data can be encoded identically."""

    focal_brand: str
    cross_brands: list[str]
    promo_cols: list[str]
    stores: list[int]
    years: list[int] | None  # None -> linear trend instead of year dummies
    tiers: list[str] | None = None  # set -> own price interacted with price tier
    # competitor prices dropped because they move in lockstep with the own price (not identified)
    collinear_brands: list[str] | None = None
    # competitor -> competitor whose price it moves with; its effect is folded into that coefficient
    absorbed: dict[str, str] | None = None

    @property
    def cross_cols(self) -> list[str]:
        return [brand_col(b) for b in self.cross_brands]


def within_store_corr(df: pd.DataFrame, a: str, b: str) -> float:
    """Correlation of two columns after removing store means (what the store-FE model sees)."""
    x = df[[a, b]] - df.groupby("store")[[a, b]].transform("mean")
    return float(x[a].corr(x[b]))


def make_spec(
    df: pd.DataFrame,
    focal_brand: str,
    brands: list[str],
    time_effects: str = "year",
    tier_interactions: bool = False,
    max_price_corr: float = 0.9,
) -> DesignSpec:
    promo_cols = [c for c in PROMO_COLS if df[c].sum() >= 30]  # drop near-empty promo types
    tiers = sorted(df["price_tier"].unique()) if tier_interactions else None
    others = [b for b in brands if b != focal_brand]
    collinear = [
        b for b in others if within_store_corr(df, "ln_price", brand_col(b)) > max_price_corr
    ]
    # competitors priced in lockstep with each other (e.g. Pepsi & Diet Pepsi) cannot be separated
    # either: keep the first, record the second as absorbed into it
    kept: list[str] = []
    absorbed: dict[str, str] = {}
    for b in (b for b in others if b not in collinear):
        twin = next((k for k in kept
                     if within_store_corr(df, brand_col(b), brand_col(k)) > max_price_corr), None)
        if twin:
            absorbed[b] = twin
        else:
            kept.append(b)
    return DesignSpec(
        focal_brand=focal_brand,
        cross_brands=kept,
        collinear_brands=collinear,
        absorbed=absorbed,
        promo_cols=promo_cols,
        stores=sorted(df["store"].unique()),
        years=sorted(df["year"].unique()) if time_effects == "year" else None,
        tiers=tiers,
    )


def design_matrix(df: pd.DataFrame, spec: DesignSpec) -> pd.DataFrame:
    """Encode ``df`` with the layout in ``spec`` (unseen stores/years map to the baseline)."""
    cols: dict[str, pd.Series] = {"const": pd.Series(1.0, index=df.index)}
    if spec.tiers:
        for t in spec.tiers:
            cols[f"ln_price_x_{t}"] = df["ln_price"] * (df["price_tier"] == t)
    else:
        cols["ln_price"] = df["ln_price"]
    for c in spec.cross_cols:
        cols[c] = df[c]
    for c in [*spec.promo_cols, *POST_COLS, *HOLIDAY_COLS, *FOURIER_COLS]:
        cols[c] = df[c].astype(float)
    if spec.years is None:
        cols["trend"] = (df["week"] - 200) / 52.0
    else:
        for y in spec.years[1:]:
            cols[f"year_{y}"] = (df["year"] == y).astype(float)
    for s in spec.stores[1:]:
        cols[f"store_{s}"] = (df["store"] == s).astype(float)
    return pd.DataFrame(cols, index=df.index)


@dataclass
class FittedModel:
    spec: DesignSpec
    result: sm.regression.linear_model.RegressionResultsWrapper
    smearing: float  # Duan smearing factor to undo the log retransformation bias

    def predict_units(self, df: pd.DataFrame) -> np.ndarray:
        X = design_matrix(df, self.spec)[self.result.params.index]
        return np.exp(X.to_numpy() @ self.result.params.to_numpy()) * self.smearing


def fit_series(
    df: pd.DataFrame,
    focal_brand: str,
    brands: list[str],
    time_effects: str = "year",
    tier_interactions: bool = False,
) -> FittedModel:
    """Fit the log-log model for one series with store-clustered standard errors."""
    spec = make_spec(df, focal_brand, brands, time_effects, tier_interactions)
    X = design_matrix(df, spec)
    X = X.loc[:, X.std() > 0].assign(const=1.0)  # drop constant columns (e.g. holiday never seen)
    y = np.log(df["units"].to_numpy(float))
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["store"].to_numpy()})
    smear = float(np.mean(np.exp(res.resid)))
    return FittedModel(spec=spec, result=res, smearing=smear)


def coef_table(model: FittedModel, series: str) -> pd.DataFrame:
    r = model.result
    ci = r.conf_int(0.05)
    return pd.DataFrame(
        {
            "series": series,
            "term": r.params.index,
            "coef": r.params.to_numpy(),
            "se": r.bse.to_numpy(),
            "ci_low": ci[0].to_numpy(),
            "ci_high": ci[1].to_numpy(),
            "p_value": r.pvalues.to_numpy(),
        }
    )

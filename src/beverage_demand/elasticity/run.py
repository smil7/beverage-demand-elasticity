"""Phase 3 runner: per-series elasticities, cross-elasticity heatmap, robustness checks.

Outputs
-------
results/elasticities.csv           own-price elasticity per series (+ CI, interpretation)
results/cross_elasticities.csv     focal series x competitor brand (same pack)
results/promo_effects.csv          promo-type lift and post-promotion dip from the regression
results/elasticity_robustness.csv  baseline vs pooled vs price-tier models
results/figures/elasticity_*.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from beverage_demand.config import results, settings
from beverage_demand.elasticity.model import (
    FOURIER_COLS,
    HOLIDAY_COLS,
    POST_COLS,
    coef_table,
    estimation_sample,
    fit_series,
)
from beverage_demand.features.panel import PROMO_LABELS, brand_col, load_panel
from beverage_demand.viz import plots
from beverage_demand.viz.plots import plt


def interpret(series: str, b: float, lo: float, hi: float) -> str:
    direction = "reduces" if b < 0 else "increases"
    return (
        f"A 1% price increase on {series} {direction} its volume by about {abs(b):.1f}% "
        f"(95% CI {lo:.2f} to {hi:.2f}); a 10% cut lifts volume by ~{(0.9**b - 1):.0%}."
    )


def fit_all(est: pd.DataFrame, brands: list[str]):
    coefs, models = [], {}
    for series, d in est.groupby("series"):
        m = fit_series(d, d["brand"].iloc[0], brands)
        models[series] = m
        coefs.append(coef_table(m, series).assign(n_obs=int(m.result.nobs), r2=m.result.rsquared))
        print(f"  {series:<26} own = {m.result.params['ln_price']:+.2f}  (n={int(m.result.nobs):,})")
    return pd.concat(coefs, ignore_index=True), models


def collinear_pairs(models: dict) -> pd.DataFrame:
    """Competitor prices that could not be separated from the own price or from another competitor."""
    rows = []
    for s, m in models.items():
        for b in m.spec.collinear_brands or []:
            rows.append({"series": s, "competitor_brand": b,
                         "note": "not identified: priced in lockstep with the focal brand"})
        for b, twin in (m.spec.absorbed or {}).items():
            rows.append({"series": s, "competitor_brand": b,
                         "note": f"not identified separately: priced in lockstep with {twin}; "
                                 f"the {twin} coefficient captures both"})
    return pd.DataFrame(rows, columns=["series", "competitor_brand", "note"])


def own_table(coefs: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    own = coefs[coefs["term"] == "ln_price"].copy()
    meta = est.groupby("series").agg(
        brand=("brand", "first"), pack=("pack", "first"), n_stores=("store", "nunique"),
        avg_price=("unit_price", "mean"), weekly_units_per_store=("units", "mean"),
    )
    own = own.merge(meta, left_on="series", right_index=True)
    own = own.rename(columns={"coef": "own_elasticity"}).drop(columns="term")
    own["sign_ok"] = own["own_elasticity"] < 0
    own["interpretation"] = [
        interpret(s, b, lo, hi)
        for s, b, lo, hi in zip(own.series, own.own_elasticity, own.ci_low, own.ci_high, strict=True)
    ]
    cols = ["series", "brand", "pack", "own_elasticity", "se", "ci_low", "ci_high", "p_value",
            "sign_ok", "n_obs", "n_stores", "r2", "avg_price", "weekly_units_per_store",
            "interpretation"]
    return own[cols].sort_values("own_elasticity").reset_index(drop=True)


def cross_table(coefs: pd.DataFrame, est: pd.DataFrame, brands: list[str],
                collinear: pd.DataFrame) -> pd.DataFrame:
    """Cross-elasticities; co-priced sister brands are listed with ``identified = False``."""
    col_to_brand = {brand_col(b): b for b in brands}
    c = coefs[coefs["term"].isin(col_to_brand)].copy()
    c["competitor_brand"] = c["term"].map(col_to_brand)
    c["identified"] = True
    c["note"] = ""
    c = pd.concat([c, collinear.assign(identified=False)], ignore_index=True)
    meta = est.groupby("series")[["brand", "pack"]].first()
    c = c.merge(meta, left_on="series", right_index=True)
    c["significant"] = (c["p_value"] < 0.05) & c["identified"]
    return c.rename(columns={"brand": "focal_brand", "coef": "cross_elasticity"})[
        ["series", "focal_brand", "pack", "competitor_brand", "cross_elasticity", "se",
         "ci_low", "ci_high", "p_value", "significant", "identified", "note"]
    ].sort_values(["pack", "focal_brand", "competitor_brand"]).reset_index(drop=True)


def promo_table(coefs: pd.DataFrame) -> pd.DataFrame:
    terms = {f"promo_{k.lower()}": v for k, v in PROMO_LABELS.items()}
    terms |= {"post_promo_1": "Week after promo", "post_promo_2": "Two weeks after promo"}
    p = coefs[coefs["term"].isin(terms)].copy()
    p["effect"] = p["term"].map(terms)
    # log-point coefficient -> % volume effect at a given price (display/feature effect)
    for c in ["coef", "ci_low", "ci_high"]:
        p[f"{c}_pct"] = np.expm1(p[c])
    return p[["series", "term", "effect", "coef", "coef_pct", "ci_low_pct", "ci_high_pct",
              "p_value"]].reset_index(drop=True)


def pooled_model(est: pd.DataFrame) -> pd.Series:
    """Robustness (a): one model for all series; own price interacted with series, shared controls."""
    d = est
    series = sorted(d["series"].unique())
    cols: dict[str, pd.Series] = {}
    for s in series:
        cols[f"lnp|{s}"] = d["ln_price"] * (d["series"] == s)
    for s in series[1:]:
        cols[f"fe|{s}"] = (d["series"] == s).astype(float)
    for c in ["promo_b", "promo_s", "promo_u", *POST_COLS, *HOLIDAY_COLS, *FOURIER_COLS]:
        cols[c] = d[c].astype(float)
    for y in sorted(d["year"].unique())[1:]:
        cols[f"y{y}"] = (d["year"] == y).astype(float)
    for st in sorted(d["store"].unique())[1:]:
        cols[f"s{st}"] = (d["store"] == st).astype(float)
    cols["const"] = pd.Series(1.0, index=d.index)
    X = pd.DataFrame(cols, index=d.index).astype(np.float32)
    res = sm.OLS(np.log(d["units"].to_numpy(float)), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["store"].to_numpy()}
    )
    return pd.Series({s: res.params[f"lnp|{s}"] for s in series}, name="pooled")


def tier_model(est: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
    """Robustness (b): own-price elasticity allowed to differ by store price tier."""
    rows = []
    for series, d in est.groupby("series"):
        m = fit_series(d, d["brand"].iloc[0], brands, tier_interactions=True)
        for t in m.spec.tiers or []:
            name = f"ln_price_x_{t}"
            if name in m.result.params:
                rows.append({"series": series, "tier": t, "elasticity": m.result.params[name],
                             "n_stores": d.loc[d["price_tier"] == t, "store"].nunique()})
    return pd.DataFrame(rows)


def plot_own(own: pd.DataFrame) -> None:
    o = own.sort_values("own_elasticity", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(o))
    colors = [plots.brand_color(s) for s in o["series"]]
    ax.barh(y, o["own_elasticity"], color=colors, height=0.6, edgecolor=plots.SURFACE, linewidth=2)
    ax.errorbar(o["own_elasticity"], y, xerr=[o["own_elasticity"] - o["ci_low"],
                o["ci_high"] - o["own_elasticity"]], fmt="none", ecolor=plots.INK_2, lw=1, capsize=3)
    ax.set_yticks(y, o["series"])
    for yi, lo, v in zip(y, o["ci_low"], o["own_elasticity"], strict=True):
        ax.text(lo - 0.1, yi, f"{v:.2f}", va="center", ha="right", fontsize=9, color=plots.INK)
    ax.axvline(-1, color=plots.INK_2, lw=1, ls="--")
    ax.text(-1, len(o) - 0.4, " unit elastic (-1)", fontsize=8, color=plots.INK_2)
    ax.set_xlabel("Own-price elasticity (% volume change for a 1% price increase)")
    ax.set_title("Every core soft-drink SKU is highly price-elastic")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(o["ci_low"].min() - 0.8, 0)
    plots.save(fig, "elasticity_own.png")


def plot_cross(cross: pd.DataFrame, brands: list[str]) -> None:
    packs = sorted(cross["pack"].unique())
    fig, axes = plt.subplots(1, len(packs), figsize=(6.6 * len(packs), 5.6),
                             gridspec_kw={"wspace": 0.45})
    vmax = float(np.nanpercentile(cross["cross_elasticity"].abs(), 95))
    for ax, pack in zip(np.atleast_1d(axes), packs, strict=True):
        m = cross[cross["pack"] == pack].pivot(index="focal_brand", columns="competitor_brand",
                                               values="cross_elasticity").reindex(index=brands, columns=brands)
        sig = cross[cross["pack"] == pack].pivot(index="focal_brand", columns="competitor_brand",
                                                 values="significant").reindex(index=brands, columns=brands)
        im = ax.imshow(m.to_numpy(float), cmap=plots.DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(brands)), brands, rotation=35, ha="right")
        ax.set_yticks(range(len(brands)), brands)
        ax.grid(False)
        for i in range(len(brands)):
            for j in range(len(brands)):
                v = m.iat[i, j]
                if i == j or np.isnan(v):
                    label = "own" if i == j else "n/i"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8, color=plots.INK_2)
                    continue
                star = "*" if sig.iat[i, j] else ""
                ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(v) > 0.65 * vmax else plots.INK)
        ax.set_title(pack)
        ax.set_xlabel("Competitor whose price changes")
        ax.set_ylabel("Brand whose volume responds")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("Cross-price elasticity (+ = substitutes)")
    fig.suptitle("Cross-price elasticities: % change in the row brand's volume when the column "
                 "brand's price rises 1%\n* = significant at 5% (store-clustered SEs); n/i = priced "
                 "in lockstep with a sister brand, effect not separately identifiable",
                 x=0.08, y=1.02, ha="left", fontsize=10.5, color=plots.INK_2)
    plots.save(fig, "elasticity_cross_heatmap.png")


def run() -> dict:
    brands = settings()["scope"]["brands"]
    est = estimation_sample(load_panel())
    print(f"Estimation sample: {len(est):,} store-weeks")
    coefs, models = fit_all(est, brands)
    coefs.to_csv(results("elasticity_coefficients.csv"), index=False)

    own = own_table(coefs, est)
    cross = cross_table(coefs, est, brands, collinear_pairs(models))
    promo = promo_table(coefs)
    own.to_csv(results("elasticities.csv"), index=False)
    cross.to_csv(results("cross_elasticities.csv"), index=False)
    promo.to_csv(results("promo_effects.csv"), index=False)

    print("Robustness: pooled model ...")
    pooled = pooled_model(est)
    print("Robustness: price-tier interactions ...")
    tiers = tier_model(est, brands)
    rob = own.set_index("series")[["own_elasticity"]].rename(columns={"own_elasticity": "baseline"})
    rob = rob.join(pooled)
    rob = rob.join(tiers.pivot(index="series", columns="tier", values="elasticity").add_prefix("tier_"))
    rob.reset_index().to_csv(results("elasticity_robustness.csv"), index=False)


    plot_own(own)
    plot_cross(cross, brands)

    ident = cross[cross["identified"]]
    pos = ident["cross_elasticity"] > 0
    summary = {
        "n_series": int(len(own)),
        "own_min": float(own["own_elasticity"].min()),
        "own_max": float(own["own_elasticity"].max()),
        "own_median": float(own["own_elasticity"].median()),
        "most_elastic": own.iloc[0]["series"],
        "least_elastic": own.iloc[-1]["series"],
        "all_own_negative": bool(own["sign_ok"].all()),
        "cross_share_positive": float(pos.mean()),
        "cross_share_positive_significant": float((pos & ident["significant"]).mean()),
        "cross_share_negative_significant": float((~pos & ident["significant"]).mean()),
        "n_cross_identified": int(len(ident)),
        "n_cross_not_identified": int((~cross["identified"]).sum()),
        "pooled_vs_baseline_corr": float(rob["baseline"].corr(rob["pooled"])),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    run()

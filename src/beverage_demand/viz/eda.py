"""Exploratory charts for notebooks/01_eda.ipynb. Each function saves a figure and returns the
numbers quoted in the notebook's "so what" captions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from beverage_demand.config import processed, settings
from beverage_demand.data.calendar import HOLIDAYS
from beverage_demand.features.panel import PROMO_LABELS
from beverage_demand.viz import plots
from beverage_demand.viz.plots import plt


def load() -> dict[str, pd.DataFrame]:
    return {
        "brand_week": pd.read_parquet(processed("brand_weekly")),
        "cal": pd.read_parquet(processed("calendar")),
        "panel": pd.read_parquet(processed("model_panel")),
    }


def volume_trend(bw: pd.DataFrame) -> pd.DataFrame:
    brands = settings()["scope"]["brands"]
    d = bw[bw["brand"].isin(brands)].pivot(index="week_start", columns="brand", values="eq_cases")
    d = d.reindex(columns=brands).rolling(13, min_periods=4).mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for b in brands:
        ax.plot(d.index, d[b], color=plots.BRAND_COLORS[b], label=b, lw=2)
        ax.text(d.index[-1], d[b].dropna().iloc[-1], f" {b}", color=plots.INK, fontsize=8.5,
                va="center")
    ax.set_title("Chain volume by brand (8-oz equivalent cases, 13-week rolling mean)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.legend(ncol=6, loc="upper left", fontsize=8.5)
    plots.save(fig, "eda_volume_trend.png")
    share = (bw.groupby("brand")["eq_cases"].sum() / bw["eq_cases"].sum()).sort_values(ascending=False)
    return share.rename("volume_share").to_frame()


def seasonality(bw: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    tot = bw.groupby("week")["eq_cases"].sum().rename("eq_cases").reset_index().merge(cal, on="week")
    # stores enter/leave the sample, so normalise by store count before comparing weeks
    n = bw.groupby("week")["n_stores"].max().rename("n_stores").reset_index()
    tot = tot.merge(n, on="week")
    tot["per_store"] = tot["eq_cases"] / tot["n_stores"]
    tot["index"] = tot["per_store"] / tot.groupby("year")["per_store"].transform("mean")
    woy = tot.groupby("week_of_year")["index"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(woy.index, woy.values, color=plots.ACCENT, lw=2)
    ax.axhline(1, color=plots.INK_2, lw=0.8, ls="--")
    hol = tot[tot["is_holiday"] == 1].groupby("holiday_name")["week_of_year"].median()
    for name, w in hol.items():
        ax.axvline(w, color=plots.GRID, lw=1, zorder=0)
        ax.text(w, ax.get_ylim()[1], name.replace("_", " "), rotation=90, fontsize=7.5,
                va="top", ha="right", color=plots.INK_2)
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Volume index (1 = year average)")
    ax.set_title("Seasonality: category volume per store by week of year")
    plots.save(fig, "eda_seasonality.png")
    lift = (tot.groupby("holiday_name")["index"].mean() - 1).drop("", errors="ignore")
    summer = woy.loc[22:35].mean() - 1
    winter = woy.loc[list(range(1, 9)) + [45, 46, 47]].mean() - 1
    return pd.DataFrame({"holiday_vs_avg": lift.reindex(HOLIDAYS)}).assign(
        summer_vs_avg=summer, winter_vs_avg=winter)


def promo_frequency(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[panel["data_gap"] == 0]
    freq = (p.groupby(["series", "promo_type"]).size().unstack(fill_value=0))
    freq = freq.div(freq.sum(axis=1), axis=0).drop(columns="none")
    freq = freq[[c for c in ["S", "B", "U", "G", "C"] if c in freq.columns]]
    depth = p[p["promo"] == 1].groupby("series")["discount"].mean()
    vol_share = (p[p["promo"] == 1].groupby("series")["units"].sum() / p.groupby("series")["units"].sum())
    order = freq.sum(axis=1).sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    left = np.zeros(len(order))
    colors = [plots.CATEGORICAL[i] for i in (0, 1, 2, 3, 4)]
    for c, col in zip(freq.columns, colors, strict=False):
        vals = freq.loc[order, c].to_numpy()
        axes[0].barh(order, vals, left=left, color=col, label=PROMO_LABELS[c],
                     edgecolor=plots.SURFACE, linewidth=1.5)
        left += vals
    for y, tot in enumerate(left):
        axes[0].text(tot + 0.01, y, f"{tot:.0%}", va="center", fontsize=8)
    axes[0].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_title("Share of store-weeks on promotion, by type")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(axis="y", visible=False)
    axes[1].barh(order, vol_share.loc[order], color=[plots.brand_color(s) for s in order])
    for y, v in enumerate(vol_share.loc[order]):
        axes[1].text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=8)
    axes[1].set_yticks([])
    axes[1].set_xlim(0, 1.1)
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_title("Share of units sold on promotion")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    plots.save(fig, "eda_promo_frequency.png")
    return pd.DataFrame({"promo_week_share": freq.sum(axis=1), "avg_depth": depth,
                         "units_on_promo_share": vol_share}).sort_values("units_on_promo_share")


def price_volume(panel: pd.DataFrame, pack: str = "2L", seed: int = 0) -> pd.DataFrame:
    p = panel[(panel["pack"] == pack) & (panel["units"] > 0) & (panel["data_gap"] == 0)]
    brands = settings()["scope"]["brands"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=False, sharey=True)
    slopes = {}
    for ax, b in zip(axes.ravel(), brands, strict=True):
        d = p[p["brand"] == b]
        # remove store size differences so the cloud shows the within-store price response
        y = np.log(d["units"]) - d.groupby("store")["units"].transform(lambda s: np.log(s).mean())
        x = d["ln_price"]
        s = d.sample(min(3000, len(d)), random_state=seed).index
        ax.scatter(np.exp(x.loc[s]), np.exp(y.loc[s]), s=4, alpha=0.25, color=plots.BRAND_COLORS[b],
                   edgecolors="none")
        slope, icept = np.polyfit(x, y, 1)
        grid = np.linspace(x.quantile(0.01), x.quantile(0.99), 50)
        ax.plot(np.exp(grid), np.exp(icept + slope * grid), color=plots.INK, lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{b}: slope {slope:.1f}", fontsize=10)
        slopes[b] = slope
    for ax in axes[1]:
        ax.set_xlabel("Unit price ($, log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Units vs. store average (log)")
    fig.suptitle(f"Price vs. volume, {pack} (store-week sample) -- a first look at elasticity",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plots.save(fig, f"eda_price_volume_{pack.replace(' ', '_')}.png")
    return pd.Series(slopes, name=f"raw_log_log_slope_{pack}").to_frame()

"""Phase 5: promotion lift, trade-spend economics and ROI.

Baseline (counterfactual) volume for a promo week = the LightGBM fold model's 1-week-ahead
prediction for that week with promotion switched off and price reset to the regular price
(``forecasting.backtest.counterfactual``). All weeks analysed are out-of-sample for that model.

Per promo week::

    lift_units           = actual units - baseline units
    discount_investment  = (regular price - promo price) * actual units
    incremental_gm       = lift_units * (regular price - unit cost)
    ROI                  = incremental_gm / discount_investment
    net_margin_impact    = incremental_gm - discount_investment

``net_margin_impact`` is exactly (actual gross margin) - (baseline gross margin at regular price),
so ROI > 1 <=> the promotion increased gross-margin dollars. Unit cost comes from the retailer's
PROFIT margin (average acquisition cost), so it already reflects any manufacturer deal money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from beverage_demand.config import processed, results, settings
from beverage_demand.features.panel import PROMO_LABELS
from beverage_demand.viz import plots
from beverage_demand.viz.plots import plt

MIN_WEEKS = 50  # minimum promo store-weeks for a promotion type to be ranked
# ROI is only meaningful when there is a real price investment; flag-only promotions (e.g. a
# bonus-buy tag with ~0% discount) have near-zero cost and an exploding ratio
MIN_AVG_DISCOUNT = 0.05


def depth_bucket(d: pd.Series, edges: list[float]) -> pd.Series:
    labels = [f"{int(a * 100)}-{int(b * 100)}%" if b < 1 else f"{int(a * 100)}%+"
              for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return pd.cut(d, edges, labels=labels, include_lowest=True, right=False).astype(str)


def promo_weeks(cf: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    p = cf[cf["promo"] == 1].copy()
    p["baseline_units"] = p["pred_baseline"]
    p["lift_units"] = p["units"] - p["baseline_units"]
    p["discount_investment"] = (p["reg_price"] - p["unit_price"]).clip(lower=0) * p["units"]
    p["unit_margin_regular"] = p["reg_price"] - p["unit_cost"]
    p["incremental_gm"] = p["lift_units"] * p["unit_margin_regular"]
    p["net_margin_impact"] = p["incremental_gm"] - p["discount_investment"]
    p["incremental_revenue"] = p["units"] * p["unit_price"] - p["baseline_units"] * p["reg_price"]
    p["depth_bucket"] = depth_bucket(p["discount"], edges)
    p["promo_label"] = p["promo_type"].map(PROMO_LABELS)
    return p


def label_events(p: pd.DataFrame) -> pd.DataFrame:
    """Contiguous promo weeks of one store x series form one promotion event."""
    p = p.sort_values(["series", "store", "week"]).copy()
    new = (p.groupby(["series", "store"])["week"].diff() != 1)
    p["event_id"] = new.cumsum()
    return p


def summarise(p: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = p.groupby(by, observed=True)
    s = g.agg(
        promo_weeks=("units", "size"),
        events=("event_id", "nunique"),
        avg_discount=("discount", "mean"),
        actual_units=("units", "sum"),
        baseline_units=("baseline_units", "sum"),
        lift_units=("lift_units", "sum"),
        discount_investment=("discount_investment", "sum"),
        incremental_gm=("incremental_gm", "sum"),
        net_margin_impact=("net_margin_impact", "sum"),
        incremental_revenue=("incremental_revenue", "sum"),
    ).reset_index()
    s["lift_pct"] = s["lift_units"] / s["baseline_units"]
    s["roi"] = s["incremental_gm"] / s["discount_investment"].where(s["discount_investment"] > 0)
    s["pays_back"] = s["roi"] > 1
    return s


def cannibalization(p: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """Volume pulled from competing in-scope brands (same pack) by each brand's price promotions.

    Uses the Phase 3 cross-elasticities: competitor j's volume changes by
    ``exp(cross_jk * ln(1 - depth)) - 1`` when brand k cuts its price by ``depth``. Pairs that are
    not identified (lockstep pricing) are skipped.
    """
    rows = []
    ident = cross[cross["identified"]]
    weekly = (p.groupby(["series", "brand", "pack"])
              .agg(depth=("discount", "mean"), lift=("lift_units", "sum"), weeks=("units", "size"))
              .reset_index())
    for _, r in weekly.iterrows():
        resp = ident[(ident["pack"] == r["pack"]) & (ident["competitor_brand"] == r["brand"])]
        effects = np.expm1(resp["cross_elasticity"] * np.log(1 - r["depth"]))
        rows.append({
            "series": r["series"], "avg_depth": r["depth"],
            "competitors_measured": len(resp),
            "avg_competitor_volume_change_pct": float(effects.mean()) if len(resp) else np.nan,
            "competitors_losing_volume": int((effects < 0).sum()),
        })
    return pd.DataFrame(rows)


def plot_lift_roi(by_type: pd.DataFrame) -> None:
    d = by_type.sort_values(["pack", "promo_label"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    packs = list(d["pack"].unique())
    types = list(d["promo_label"].unique())
    width = 0.8 / len(packs)
    pack_colors = {pk: c for pk, c in zip(packs, [plots.ACCENT, plots.CONTRAST], strict=False)}
    for ax, metric, title in [
        (axes[0], "lift_pct", "Volume lift vs. no-promo baseline"),
        (axes[1], "roi", "ROI = incremental margin / discount cost"),
    ]:
        for j, pk in enumerate(packs):
            sub = d[d["pack"] == pk].set_index("promo_label").reindex(types)
            x = np.arange(len(types)) + j * width
            vals = sub[metric].to_numpy(float)
            ax.bar(x, vals, width=width * 0.92, color=pack_colors[pk], label=pk,
                   edgecolor=plots.SURFACE, linewidth=1.5)
            for xi, v in zip(x, vals, strict=True):
                if np.isfinite(v):
                    txt = f"{v:.0%}" if metric == "lift_pct" else f"{v:.2f}"
                    ax.text(xi, v, txt, ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.set_xticks(np.arange(len(types)) + width * (len(packs) - 1) / 2,
                      [t.replace(" (", "\n(") for t in types], fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=plots.INK_2, lw=0.8)
    axes[1].axhline(1, color=plots.INK_2, lw=1, ls="--")
    axes[1].text(len(types) - 0.6, 1.02, "break-even", fontsize=8, color=plots.INK_2, ha="right")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].legend(title="Pack", loc="upper left")
    fig.suptitle("Promotions lift volume a lot, but few pay back their discount cost",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plots.save(fig, "promo_lift_roi_by_type.png")


def plot_depth(by_depth: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for pk, color in zip(sorted(by_depth["pack"].unique()), [plots.ACCENT, plots.CONTRAST], strict=False):
        d = by_depth[by_depth["pack"] == pk]
        ax.plot(d["depth_bucket"], d["roi"], marker="o", color=color, label=pk, ms=8)
        for x, v in zip(d["depth_bucket"], d["roi"], strict=True):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
    ax.axhline(1, color=plots.INK_2, lw=1, ls="--")
    ax.set_xlabel("Discount depth vs. regular price")
    ax.set_ylabel("ROI (incremental margin / discount cost)")
    ax.set_title("Promotion ROI by discount depth")
    ax.legend(title="Pack")
    plots.save(fig, "promo_roi_by_depth.png")


def run() -> dict:
    cfg = settings()
    cf = pd.read_parquet(processed("promo_counterfactual"))
    cross = pd.read_csv(results("cross_elasticities.csv"))
    p = label_events(promo_weeks(cf, cfg["promo"]["depth_buckets"]))
    p.to_parquet(processed("fact_promo_events"), index=False)

    ranking = summarise(p, ["brand", "pack", "promo_label", "depth_bucket"])
    ranking = ranking[(ranking["promo_weeks"] >= MIN_WEEKS)
                      & (ranking["avg_discount"] >= MIN_AVG_DISCOUNT)]
    ranking = ranking.sort_values("roi", ascending=False).reset_index(drop=True)
    ranking.to_csv(results("promo_roi.csv"), index=False)
    by_type = summarise(p, ["pack", "promo_label"])
    by_type.to_csv(results("promo_by_type.csv"), index=False)
    by_depth = summarise(p, ["pack", "depth_bucket"])
    by_depth.to_csv(results("promo_by_depth.csv"), index=False)
    by_series = summarise(p, ["series"])
    cann = cannibalization(p, cross)
    by_series = by_series.merge(cann, on="series", how="left")
    by_series.to_csv(results("promo_by_series.csv"), index=False)
    overall = summarise(p.assign(all="all"), ["all"]).iloc[0]
    by_pack = summarise(p, ["pack"]).set_index("pack")
    effects = pd.read_csv(results("promo_effects.csv"))
    effects["pack"] = effects["series"].str.split(" | ", regex=False).str[1]
    dip = effects[effects["term"] == "post_promo_1"].groupby("pack")["coef_pct"].median()

    plot_lift_roi(by_type)
    plot_depth(by_depth)

    fmt = ranking.assign(
        promotion=lambda d: d["brand"] + " " + d["pack"] + " - " + d["promo_label"] + ", "
        + d["depth_bucket"] + " off",
        lift=lambda d: d["lift_pct"].map("{:+.0%}".format),
        ROI=lambda d: d["roi"].map("{:.2f}".format),
        net=lambda d: d["net_margin_impact"].map("${:,.0f}".format),
    )
    cols = ["promotion", "promo_weeks", "lift", "ROI", "net"]
    md = ["# Promotion ranking by ROI", "",
          "_Generated by `beverage_demand.promo.analysis`. Out-of-sample backtest period "
          f"(weeks {cf['week'].min()}-{cf['week'].max()}); promotion types with at least "
          f"{MIN_WEEKS} promoted store-weeks and an average discount of at least "
          f"{MIN_AVG_DISCOUNT:.0%}; `lift` = volume vs. the no-promo baseline; "
          "`net` = gross-margin change vs. not promoting._", "",
          "## Top 10", "", fmt.head(10)[cols].to_markdown(index=False), "",
          "## Bottom 10", "", fmt.tail(10)[cols].to_markdown(index=False), ""]
    results("promo_ranking.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "promo_weeks": int(overall["promo_weeks"]),
        "promo_events": int(p["event_id"].nunique()),
        "avg_lift_pct": float(overall["lift_pct"]),
        "lift_pct_by_pack": {k: float(v) for k, v in by_pack["lift_pct"].items()},
        "roi_by_pack": {k: float(v) for k, v in by_pack["roi"].items()},
        "post_promo_dip_week1_median_by_pack": {k: float(v) for k, v in dip.items()},
        "overall_roi": float(overall["roi"]),
        "total_discount_investment": float(overall["discount_investment"]),
        "total_net_margin_impact": float(overall["net_margin_impact"]),
        "n_ranked_types": int(len(ranking)),
        "share_types_roi_gt_1": float(ranking["pays_back"].mean()),
        "best_type": fmt.iloc[0]["promotion"],
        "best_type_roi": float(ranking.iloc[0]["roi"]),
        "worst_type": fmt.iloc[-1]["promotion"],
        "worst_type_roi": float(ranking.iloc[-1]["roi"]),
    }
    print(by_type.to_string(index=False))
    print(summary)
    return summary


if __name__ == "__main__":
    run()

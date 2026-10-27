"""Promotion lift, discount investment and ROI.

The economics are computed in SQL (``sql/analysis/promo_events.sql`` and ``promo_roi.sql``) on
the ``promo_counterfactual`` table written by the forecast backtest: for every promoted store-week
it holds the actual units and the model's prediction of what would have sold *without* the
promotion (all out-of-sample). ROI = incremental gross margin / discount investment, so ROI > 1
means the promotion added gross-margin dollars.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from beverage_demand.config import results
from beverage_demand.db import connect, query_file, run_sql_file, write_table
from beverage_demand.viz import plots
from beverage_demand.viz.plots import plt

MIN_WEEKS = 50  # minimum promo store-weeks for a promotion type to be ranked
# ROI is only meaningful when there is a real price investment; flag-only promotions (e.g. a
# bonus-buy tag with ~0% discount) have near-zero cost and an exploding ratio
MIN_AVG_DISCOUNT = 0.05


def roi_by(con: duckdb.DuckDBPyConnection, group_by: str, min_weeks: int = 0,
           min_discount: float = 0.0) -> pd.DataFrame:
    """Aggregate ``promo_events`` with ``sql/analysis/promo_roi.sql``."""
    return query_file(con, "analysis/promo_roi.sql", group_by=group_by, min_weeks=min_weeks,
                      min_discount=min_discount)


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
    cross = pd.read_csv(results("cross_elasticities.csv"))
    with connect() as con:
        run_sql_file(con, "analysis/promo_events.sql")
        p = con.execute("SELECT * FROM promo_events").df()
        weeks = con.execute("SELECT MIN(week), MAX(week) FROM promo_events").fetchone()
        ranking = roi_by(con, "brand, pack, promo_label, depth_bucket", MIN_WEEKS, MIN_AVG_DISCOUNT)
        ranking = ranking.sort_values("roi", ascending=False).reset_index(drop=True)
        by_type = roi_by(con, "pack, promo_label")
        by_depth = roi_by(con, "pack, depth_bucket")
        by_pack = roi_by(con, "pack").set_index("pack")
        by_series = roi_by(con, "series").merge(cannibalization(p, cross), on="series", how="left")
        overall = roi_by(con, "promo_type IS NOT NULL").iloc[0]  # one group = all promo weeks
        for name, df in [("promo_roi", ranking), ("promo_by_type", by_type),
                         ("promo_by_depth", by_depth), ("promo_by_series", by_series)]:
            write_table(con, name, df)
            df.to_csv(results(f"{name}.csv"), index=False)

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
          "_Generated by `beverage_demand.promo.analysis` (SQL: `sql/analysis/promo_roi.sql`). "
          f"Out-of-sample backtest period (weeks {weeks[0]}-{weeks[1]}); promotion types with at "
          f"least {MIN_WEEKS} promoted store-weeks and an average discount of at least "
          f"{MIN_AVG_DISCOUNT:.0%}; `lift` = volume vs. the no-promo baseline; "
          "`net` = gross-margin change vs. not promoting._", "",
          "## Top 10", "", fmt.head(10)[cols].to_markdown(index=False), "",
          "## Bottom 10", "", fmt.tail(10)[cols].to_markdown(index=False), ""]
    results("promo_ranking.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "promo_weeks": int(overall["promo_weeks"]),
        "promo_events": int(overall["events"]),
        "avg_lift_pct": float(overall["lift_pct"]),
        "lift_pct_by_pack": {k: float(v) for k, v in by_pack["lift_pct"].items()},
        "roi_by_pack": {k: float(v) for k, v in by_pack["roi"].items()},
        "post_promo_dip_week1_median_by_pack": {k: float(v) for k, v in dip.items()},
        "overall_roi": float(overall["roi"]),
        "n_ranked_types": int(len(ranking)),
        "share_types_roi_gt_1": float(ranking["pays_back"].mean()),
    }
    print(by_type.to_string(index=False))
    print(summary)
    return summary


if __name__ == "__main__":
    run()

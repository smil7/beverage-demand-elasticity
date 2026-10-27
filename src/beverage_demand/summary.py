"""Build ``results/summary.json``: every headline number in the README comes from here."""

from __future__ import annotations

import json

import pandas as pd

from beverage_demand.config import processed, results, settings


def _r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def build() -> dict:
    counts = pd.read_parquet(processed("_ingest_counts")).iloc[0]
    fact = pd.read_parquet(processed("fact_sales_weekly"), columns=["store", "week", "brand", "pack"])
    panel = pd.read_parquet(processed("model_panel"), columns=["series", "store", "week"])
    cal = pd.read_parquet(processed("dim_calendar"))

    own = pd.read_csv(results("elasticities.csv"))
    cross = pd.read_csv(results("cross_elasticities.csv"))
    rob = pd.read_csv(results("elasticity_robustness.csv"))
    effects = pd.read_csv(results("promo_effects.csv"))
    fm = pd.read_csv(results("forecast_metrics.csv"))
    by_pack_depth = pd.read_csv(results("promo_by_depth.csv"))
    ranking = pd.read_csv(results("promo_roi.csv"))
    by_type = pd.read_csv(results("promo_by_type.csv"))
    events = pd.read_parquet(processed("fact_promo_events"))
    opt = pd.read_csv(results("price_optimization.csv"))

    wk = cal.set_index("week")
    data = {
        "raw_movement_rows": int(counts["raw_rows"]),
        "clean_movement_rows": int(counts["clean_rows"]),
        "fact_rows_brand_pack_store_week": int(len(fact)),
        "stores": int(fact["store"].nunique()),
        "weeks_with_data": int(fact["week"].nunique()),
        "first_week_start": str(wk.loc[fact["week"].min(), "week_start"].date()),
        "last_week_end": str(wk.loc[fact["week"].max(), "week_end"].date()),
        "modelled_series_brand_pack": int(panel["series"].nunique()),
        "modelled_store_series": int(panel.groupby(["series", "store"]).ngroups),
        "model_panel_rows": int(len(panel)),
    }

    ident = cross[cross["identified"]]
    elasticity = {
        "own_price_elasticity_min": _r(own["own_elasticity"].min(), 2),
        "own_price_elasticity_max": _r(own["own_elasticity"].max(), 2),
        "own_price_elasticity_median": _r(own["own_elasticity"].median(), 2),
        "most_elastic_series": own.sort_values("own_elasticity").iloc[0]["series"],
        "least_elastic_series": own.sort_values("own_elasticity").iloc[-1]["series"],
        "median_elasticity_24pk_can": _r(own.loc[own["pack"] == "24pk can", "own_elasticity"].median(), 2),
        "median_elasticity_2L": _r(own.loc[own["pack"] == "2L", "own_elasticity"].median(), 2),
        "all_own_elasticities_negative": bool((own["own_elasticity"] < 0).all()),
        "cross_pairs_identified": int(len(ident)),
        "cross_pairs_not_identified_lockstep_pricing": int((~cross["identified"]).sum()),
        "cross_share_positive": _r((ident["cross_elasticity"] > 0).mean(), 3),
        "cross_share_positive_and_significant": _r(
            ((ident["cross_elasticity"] > 0) & ident["significant"]).mean(), 3),
        "pooled_vs_per_series_correlation": _r(rob["baseline"].corr(rob["pooled"]), 3),
        "max_abs_tier_deviation": _r(
            rob[[c for c in rob.columns if c.startswith("tier_")]].sub(rob["baseline"], axis=0)
            .abs().max().max(), 2),
    }

    def w(slice_: str, group: str, model: str) -> float:
        row = fm[(fm["slice"] == slice_) & (fm["group"] == group) & (fm["model"] == model)]
        return float(row["wape"].iloc[0])

    def m(slice_: str, group: str, model: str, col: str) -> float:
        row = fm[(fm["slice"] == slice_) & (fm["group"] == group) & (fm["model"] == model)]
        return float(row[col].iloc[0])

    baselines = ["seasonal_naive", "moving_avg_4", "regression"]
    fc = settings()["forecast"]
    forecast = {"backtest_folds": fc["n_folds"], "horizon_weeks": fc["horizon"],
                "test_weeks": f"{fc['first_origin'] + 1}-{fc['first_origin'] + fc['n_folds'] * fc['fold_step']}"}
    for name, (sl, grp) in {"overall": ("overall", "all"), "promo_weeks": ("promo", "promo weeks"),
                            "non_promo_weeks": ("promo", "non-promo weeks")}.items():
        lg = w(sl, grp, "lightgbm")
        sn = w(sl, grp, "seasonal_naive")
        best = min(baselines, key=lambda b, sl=sl, grp=grp: w(sl, grp, b))
        forecast[name] = {
            "wape_lightgbm": _r(lg), "wape_seasonal_naive": _r(sn),
            "wape_moving_avg_4": _r(w(sl, grp, "moving_avg_4")),
            "wape_regression": _r(w(sl, grp, "regression")),
            "best_baseline": best, "wape_best_baseline": _r(w(sl, grp, best)),
            "wape_improvement_vs_seasonal_naive_pct": _r((1 - lg / sn) * 100, 1),
            "wape_improvement_vs_best_baseline_pct": _r((1 - lg / w(sl, grp, best)) * 100, 1),
            "mape_lightgbm": _r(m(sl, grp, "lightgbm", "mape")),
            "mape_seasonal_naive": _r(m(sl, grp, "seasonal_naive", "mape")),
            "mape_improvement_vs_seasonal_naive_pct": _r(
                (1 - m(sl, grp, "lightgbm", "mape") / m(sl, grp, "seasonal_naive", "mape")) * 100, 1),
            "bias_lightgbm": _r(m(sl, grp, "lightgbm", "bias")),
            "test_rows": int(m(sl, grp, "lightgbm", "n")),
        }
    forecast["chain_level"] = {
        "wape_lightgbm": _r(w("chain_level", "all series x week", "lightgbm")),
        "wape_seasonal_naive": _r(w("chain_level", "all series x week", "seasonal_naive")),
        "wape_regression": _r(w("chain_level", "all series x week", "regression")),
    }
    shap = pd.read_csv(results("forecast_shap_importance.csv"))
    forecast["top_shap_features"] = shap.iloc[:5, 0].tolist()

    tot = events[["baseline_units", "lift_units", "incremental_gm", "discount_investment",
                  "net_margin_impact"]].sum()
    deep = by_pack_depth[by_pack_depth["depth_bucket"] == "30%+"]
    mid = by_pack_depth[by_pack_depth["depth_bucket"] == "10-20%"]
    dip = effects[effects["term"] == "post_promo_1"].assign(
        pack=lambda d: d["series"].str.split(" | ", regex=False).str[1])
    promo = {
        "promo_store_weeks_analysed": int(len(events)),
        "promo_events": int(events["event_id"].nunique()),
        "avg_lift_pct": _r(tot["lift_units"] / tot["baseline_units"] * 100, 1),
        "avg_lift_pct_24pk_can": _r(_lift(events, "24pk can") * 100, 1),
        "avg_lift_pct_2L": _r(_lift(events, "2L") * 100, 1),
        "overall_roi": _r(tot["incremental_gm"] / tot["discount_investment"], 3),
        "total_discount_investment_usd": _r(tot["discount_investment"], 0),
        "total_net_gross_margin_impact_usd": _r(tot["net_margin_impact"], 0),
        "promo_types_ranked": int(len(ranking)),
        "share_promo_types_roi_gt_1": _r(ranking["pays_back"].mean(), 3),
        "best_promo_type": _label(ranking.iloc[0]),
        "best_promo_type_roi": _r(ranking.iloc[0]["roi"], 2),
        "largest_margin_gain_promo_type": _label(ranking.sort_values("net_margin_impact").iloc[-1]),
        "largest_margin_gain_roi": _r(ranking.sort_values("net_margin_impact").iloc[-1]["roi"], 2),
        "largest_margin_gain_usd": _r(ranking["net_margin_impact"].max(), 0),
        "worst_promo_type": _label(ranking.iloc[-1]),
        "worst_promo_type_roi": _r(ranking.iloc[-1]["roi"], 2),
        "roi_by_depth": {f"{r.pack} {r.depth_bucket}": _r(r.roi, 2) for r in by_pack_depth.itertuples()},
        "roi_10_20pct_discounts": {r.pack: _r(r.roi, 2) for r in mid.itertuples()},
        "roi_30pct_plus_discounts": {r.pack: _r(r.roi, 2) for r in deep.itertuples()},
        "share_of_discount_spend_on_30pct_plus": _r(
            deep["discount_investment"].sum() / by_pack_depth["discount_investment"].sum(), 3),
        "roi_by_type": {f"{r.pack} {r.promo_label}": _r(r.roi, 2) for r in by_type.itertuples()},
        "post_promo_dip_week1_median_pct": {
            k: _r(v * 100, 1) for k, v in dip.groupby("pack")["coef_pct"].median().items()},
    }

    ex = opt[opt["series"] == "Coke | 2L"].iloc[0]
    optimisation = {
        "example_series": ex["series"],
        "regular_price": _r(ex["regular_price"], 2),
        "unit_cost": _r(ex["unit_cost"], 2),
        "elasticity": _r(ex["elasticity"], 2),
        "optimal_price": _r(ex["optimal_price"], 2),
        "price_change_pct": _r(ex["price_change_pct"] * 100, 1),
        "volume_change_pct": _r(ex["volume_change_pct"] * 100, 1),
        "margin_uplift_pct": _r(ex["margin_uplift_pct"] * 100, 1),
        "optimal_price_range_from_elasticity_ci": [_r(ex["optimal_price_range_low"], 2),
                                                   _r(ex["optimal_price_range_high"], 2)],
        "assumptions": "constant elasticity, competitors do not react, unit cost fixed, "
                       "+/-20% price bounds",
    }
    summary = {"data": data, "elasticity": elasticity, "forecast": forecast, "promo": promo,
               "price_optimisation": optimisation}
    results("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _lift(events: pd.DataFrame, pack: str) -> float:
    e = events[events["pack"] == pack]
    return e["lift_units"].sum() / e["baseline_units"].sum()


def _label(r: pd.Series) -> str:
    return f"{r['brand']} {r['pack']} - {r['promo_label']}, {r['depth_bucket']} off"


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))

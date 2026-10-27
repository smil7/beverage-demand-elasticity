"""Build ``results/summary.json``: every headline number in the README comes from here."""

from __future__ import annotations

import json

import pandas as pd

from beverage_demand.config import results, settings
from beverage_demand.db import connect


def _r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def build() -> dict:
    with connect(read_only=True) as con:
        counts = con.execute(
            "SELECT * FROM etl_run_log ORDER BY loaded_at DESC LIMIT 1").df().iloc[0]
        facts = con.execute("""
            SELECT COUNT(*) AS n, COUNT(DISTINCT store) AS stores, COUNT(DISTINCT week) AS weeks,
                   MIN(c.week_start) AS first_day, MAX(c.week_end) AS last_day
            FROM fact_sales_weekly f JOIN dim_calendar c USING (week)""").df().iloc[0]
        panel = con.execute("""
            SELECT COUNT(*) AS n, COUNT(DISTINCT series) AS series,
                   COUNT(DISTINCT (series, store)) AS store_series FROM model_panel""").df().iloc[0]
        tables = con.execute("""
            SELECT table_name, estimated_size FROM duckdb_tables() ORDER BY table_name""").df()
        events = con.execute("SELECT * FROM promo_events").df()
        trials = con.execute("SELECT * FROM tuning_trials ORDER BY cv_wape").df()
        ablation = con.execute("SELECT * FROM ablation_results").df()

    own = pd.read_csv(results("elasticities.csv"))
    cross = pd.read_csv(results("cross_elasticities.csv"))
    rob = pd.read_csv(results("elasticity_robustness.csv"))
    effects = pd.read_csv(results("promo_effects.csv"))
    fm = pd.read_csv(results("forecast_metrics.csv"))
    by_pack_depth = pd.read_csv(results("promo_by_depth.csv"))
    ranking = pd.read_csv(results("promo_roi.csv"))
    by_type = pd.read_csv(results("promo_by_type.csv"))

    data = {
        "raw_movement_rows": int(counts["raw_rows"]),
        "clean_movement_rows": int(counts["clean_rows"]),
        "fact_rows_brand_pack_store_week": int(facts["n"]),
        "stores": int(facts["stores"]),
        "weeks_with_data": int(facts["weeks"]),
        "first_week_start": str(pd.Timestamp(facts["first_day"]).date()),
        "last_week_end": str(pd.Timestamp(facts["last_day"]).date()),
        "modelled_series_brand_pack": int(panel["series"]),
        "modelled_store_series": int(panel["store_series"]),
        "model_panel_rows": int(panel["n"]),
        "warehouse_tables": int(len(tables)),
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

    baselines = ["seasonal_naive", "moving_avg_4", "regression"]  # hist_gbm is reported separately
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
            "wape_hist_gbm": _r(w(sl, grp, "hist_gbm")),
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
    best = json.loads(results("best_params.json").read_text(encoding="utf-8"))
    full = ablation.iloc[0]["wape"]
    modelling = {
        "tuning_trials": int(len(trials)),
        "tuning_cv_folds": best["cv_folds"],
        "best_params": best["params"],
        "best_cv_wape": _r(best["cv_wape"]),
        "worst_cv_wape": _r(trials["cv_wape"].max()),
        "boosting_rounds": best["rounds"],
        "lightgbm_vs_hist_gbm_wape": [_r(w("overall", "all", "lightgbm")),
                                      _r(w("overall", "all", "hist_gbm"))],
        "ablation_wape_increase": {
            r.variant.replace("without ", ""): _r(r.wape - full) for r in ablation.iloc[1:].itertuples()},
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

    summary = {"data": data, "modelling": modelling, "elasticity": elasticity,
               "forecast": forecast, "promo": promo}
    results("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _lift(events: pd.DataFrame, pack: str) -> float:
    e = events[events["pack"] == pack]
    return e["lift_units"].sum() / e["baseline_units"].sum()


def _label(r: pd.Series) -> str:
    return f"{r['brand']} {r['pack']} - {r['promo_label']}, {r['depth_bucket']} off"


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))

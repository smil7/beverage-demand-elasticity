"""Phase 7: export a Power BI star schema to ``powerbi/export/`` (git-ignored: derived from licensed data).

Tables
------
fact_sales_weekly   brand x pack x store x week sales (Parquet: ~2.4M rows)
fact_forecast       backtest forecasts vs actuals (one row per series x store x week x model)
fact_promo_events   promoted store-weeks with baseline, lift, investment and ROI inputs
dim_product         brand x pack (one row per series; UPC-level dim kept as dim_upc)
dim_store, dim_calendar
elasticities, cross_elasticities, promo_roi, price_optimization
scenario_grid       pre-computed -20%..+20% price scenarios per series (what-if page)
"""

from __future__ import annotations

import shutil

import pandas as pd

from beverage_demand.config import path, processed, results


def run() -> None:
    out = path("powerbi_export_dir")

    fact = pd.read_parquet(processed("fact_sales_weekly"))
    fact["series"] = fact["brand"] + " | " + fact["pack"]
    fact["gross_margin"] = fact["dollars"] - fact["cogs"]
    fact["promo_flag"] = fact[["any_sale_b", "any_sale_s", "any_sale_c", "any_sale_g"]].max(axis=1)
    fact.to_parquet(out / "fact_sales_weekly.parquet", index=False)

    pred = pd.read_parquet(processed("backtest_predictions"))
    long = pred.melt(id_vars=["series", "store", "week", "h", "fold", "units", "promo", "promo_type"],
                     value_vars=["lightgbm", "seasonal_naive", "moving_avg_4", "regression"],
                     var_name="model", value_name="forecast_units")
    long = long.rename(columns={"units": "actual_units"})
    long["abs_error"] = (long["actual_units"] - long["forecast_units"]).abs()
    long.to_csv(out / "fact_forecast.csv", index=False)

    ev = pd.read_parquet(processed("fact_promo_events"))
    ev[["series", "brand", "pack", "store", "week", "event_id", "promo_type", "promo_label",
        "depth_bucket", "discount", "unit_price", "reg_price", "unit_cost", "units",
        "baseline_units", "lift_units", "discount_investment", "incremental_gm",
        "net_margin_impact", "incremental_revenue"]].to_csv(out / "fact_promo_events.csv", index=False)

    upc = pd.read_parquet(processed("dim_product"))
    upc.to_csv(out / "dim_upc.csv", index=False)
    prod = (fact.groupby(["series", "brand", "pack"]).agg(total_dollars=("dollars", "sum"))
            .reset_index())
    prod["in_model_scope"] = prod["series"].isin(pred["series"].unique())
    prod.to_csv(out / "dim_product.csv", index=False)
    pd.read_parquet(processed("dim_store")).to_csv(out / "dim_store.csv", index=False)
    pd.read_parquet(processed("dim_calendar")).to_csv(out / "dim_calendar.csv", index=False)

    for name in ["elasticities", "cross_elasticities", "promo_roi", "price_optimization",
                 "forecast_metrics"]:
        shutil.copy(results(f"{name}.csv"), out / f"{name}.csv")

    from beverage_demand.scenario.simulate import scenario_grid

    scenario_grid().to_csv(out / "scenario_grid.csv", index=False)
    print(f"Power BI export written to {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<28} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    run()

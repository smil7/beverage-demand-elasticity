"""Command-line entry points.

    uv run bev <step>     # warehouse | quality | features | elasticity | tune | forecast | ablation
                          # | promo | summary | all
    uv run scenario --brand "Coke" --pack 2L --price-change -10 --weeks 4 [--promo S]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def step_warehouse() -> None:
    from beverage_demand.ingest import warehouse

    warehouse.run()


def step_quality() -> None:
    from beverage_demand.ingest import quality

    quality.run()


def step_features() -> None:
    from beverage_demand.features import panel

    panel.run()


def step_elasticity() -> None:
    from beverage_demand.elasticity import run

    run.run()


def step_tune() -> None:
    from beverage_demand.forecasting import tuning

    tuning.run()


def step_forecast() -> None:
    from beverage_demand.forecasting import backtest, report

    backtest.run()
    report.run()


def step_ablation() -> None:
    from beverage_demand.forecasting import ablation

    ablation.run()


def step_promo() -> None:
    from beverage_demand.promo import analysis

    analysis.run()


def step_summary() -> None:
    from beverage_demand import summary

    summary.build()
    print("wrote results/summary.json")


STEPS = {
    "warehouse": step_warehouse,
    "quality": step_quality,
    "features": step_features,
    "elasticity": step_elasticity,
    "tune": step_tune,
    "forecast": step_forecast,
    "ablation": step_ablation,
    "promo": step_promo,
    "summary": step_summary,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bev", description="Run pipeline steps")
    ap.add_argument("step", choices=[*STEPS, "all"])
    args = ap.parse_args(argv)
    steps = list(STEPS) if args.step == "all" else [args.step]
    for s in steps:
        print(f"\n=== {s} ===")
        STEPS[s]()
    return 0


def format_scenario(df: pd.DataFrame) -> str:
    t = df.copy()
    out = pd.DataFrame({
        "Series": t["series"],
        "Role": t["role"],
        "Units (base)": t["units_base"].map("{:,.0f}".format),
        "Units (scenario)": t["units_scenario"].map("{:,.0f}".format),
        "Units 90% range": [f"{lo:,.0f}-{hi:,.0f}" for lo, hi in zip(t["units_p5"], t["units_p95"], strict=True)],
        "Volume": t["units_change_pct"].map("{:+.1%}".format),
        "Revenue": t["revenue_change_pct"].map("{:+.1%}".format),
        "Gross margin": t["gm_change_pct"].map("{:+.1%}".format),
        "GM $ change": (t["gm_scenario"] - t["gm_base"]).map("{:+,.0f}".format),
    })
    tgt = t.iloc[0]
    head = (f"Scenario: {tgt['series']} regular price ${tgt['price_base']:.2f} -> "
            f"${tgt['price_scenario']:.2f}\n"
            "Assumes constant elasticities and no competitor price reaction; "
            "ranges from elasticity standard errors.\n")
    return head + out.to_string(index=False)


def scenario_main(argv: list[str] | None = None) -> int:
    from beverage_demand.scenario.simulate import simulate

    ap = argparse.ArgumentParser(prog="scenario", description="What-if price/promo scenario")
    ap.add_argument("--brand", required=True, help='e.g. "Coke", "Diet Pepsi", "Private Label"')
    ap.add_argument("--pack", required=True, help='"2L" or "24pk can"')
    ap.add_argument("--price-change", type=float, required=True, help="percent, e.g. -10")
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--promo", choices=["B", "S", "C", "G", "U"], default=None,
                    help="optional promotion type run alongside the price change")
    ap.add_argument("--stores", type=int, nargs="*", default=None, help="store numbers (default all)")
    a = ap.parse_args(argv)
    try:
        df = simulate(a.brand, a.pack, a.price_change, promo_type=a.promo, weeks=a.weeks,
                      stores=tuple(a.stores) if a.stores else None)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_scenario(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the whole analysis in order (about 15 minutes on a laptop).

Usage: uv run python scripts/run_all.py
Needs the raw files in data/raw (run scripts/download_data.py first).
"""

from beverage_demand import summary
from beverage_demand.data import load
from beverage_demand.elasticity import run as elasticity
from beverage_demand.features import panel
from beverage_demand.forecasting import backtest, report, tuning
from beverage_demand.promo import analysis as promo

STEPS = [
    ("load and clean the raw data", load.run),
    ("build the modelling panel", panel.run),
    ("price elasticity models", elasticity.run),
    ("tune LightGBM (time-series CV)", tuning.run),
    ("backtest LightGBM vs baselines", backtest.run),
    ("forecast charts and SHAP", report.run),
    ("promotion lift and ROI", promo.run),
    ("results/summary.json", summary.build),
]

if __name__ == "__main__":
    for name, step in STEPS:
        print(f"\n=== {name} ===")
        step()

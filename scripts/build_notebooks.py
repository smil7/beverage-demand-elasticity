"""Generate the narrative notebooks in notebooks/ (logic lives in src/; notebooks only narrate).

Usage: uv run python scripts/build_notebooks.py && make notebooks
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """import pandas as pd
from IPython.display import Image, Markdown, display
from beverage_demand.config import results, figure
from beverage_demand.db import connect
con = connect(read_only=True)  # the DuckDB warehouse: data/warehouse.duckdb
pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def fig(name: str):
    return code(f'Image(filename=str(figure("{name}")))')


NOTEBOOKS = {
    "01_eda.ipynb": [
        md("""
# 01 -- Exploratory analysis: what does soft-drink demand look like?

Dominick's Finer Foods (Chicago, 93 stores, Sep 1989 - May 1997), soft-drink category.
All charts are produced by `beverage_demand.viz.eda`; this notebook adds the commercial reading.
Each chart has a **So what** line written for a non-technical commercial audience.
"""),
        code(SETUP + "\nfrom beverage_demand.viz import eda\nd = eda.load()"),
        md("""
## 0. What is in the warehouse

Everything lives in one DuckDB database built by `sql/warehouse/*.sql`. A few queries to get a
feel for the data before any modelling.
"""),
        code('''con.sql("""
SELECT table_name, estimated_size AS rows
FROM duckdb_tables()
WHERE table_name IN ('stg_movement', 'fact_sales_weekly', 'fact_brand_week',
                     'dim_product', 'dim_store', 'dim_calendar', 'model_panel')
ORDER BY estimated_size DESC
""").df()'''),
        code('''# top brand x pack combinations by revenue, and how much of their volume sells on promotion
con.sql("""
SELECT brand, pack,
       ROUND(SUM(dollars) / 1e6, 1)                                   AS revenue_musd,
       SUM(units)                                                     AS units,
       ROUND(SUM(units_sale_b + units_sale_s + units_sale_c + units_sale_g)
             / SUM(units), 3)                                         AS share_units_with_sale_code
FROM fact_sales_weekly
WHERE brand <> 'Other'
GROUP BY brand, pack
ORDER BY revenue_musd DESC
LIMIT 12
""").df()'''),
        code('''# year-over-year revenue growth per brand with a window function
con.sql("""
WITH yearly AS (
    SELECT f.brand, c.year, SUM(f.dollars) AS revenue
    FROM fact_sales_weekly f JOIN dim_calendar c USING (week)
    WHERE c.year BETWEEN 1990 AND 1996 AND f.brand IN ('Coke', 'Pepsi', 'Private Label')
    GROUP BY f.brand, c.year
)
SELECT brand, year, ROUND(revenue / 1e6, 2) AS revenue_musd,
       ROUND(revenue / LAG(revenue) OVER (PARTITION BY brand ORDER BY year) - 1, 3) AS yoy_growth
FROM yearly ORDER BY brand, year
""").df()'''),
        md("""
**So what:** Coke and Pepsi in 24-packs and 2-litres are the money makers, and for them the
majority of units carry a sale code -- before even looking at prices, the data say this is a
promotion-driven category.
"""),
        md("## 1. Volume trend by brand"),
        code('share = eda.volume_trend(d["brand_week"])\nshare.head(10)'),
        fig("eda_volume_trend.png"),
        md("""
**So what:** Pepsi is the biggest single brand (about 17% of category volume vs. 12% for Coke) and
grew steadily into 1997, while Dominick's private label lost ground after 1995. Brand-level
strategy matters: the same price move will not play out the same way for every brand.
"""),
        md("## 2. Seasonality and holidays"),
        code('season = eda.seasonality(d["brand_week"], d["cal"])\nseason'),
        fig("eda_seasonality.png"),
        md("""
**So what:** Summer holidays are the category's peak weeks -- July 4th weeks run roughly 37% above
an average week, Labor Day about 32% and Memorial Day about 27%, with Christmas/New Year close
behind. Thanksgiving and Presidents' Day weeks are *below* average. Promotion calendars and
inventory plans should be built around these weeks, and forecasts must know about them.
"""),
        md("## 3. How often, how deep, and how much volume is promoted"),
        code('promo = eda.promo_frequency(d["panel"])\npromo'),
        fig("eda_promo_frequency.png"),
        md("""
**So what:** This is a promotion-driven category: the core brands are on some kind of deal in
43-66% of store-weeks, at an average depth of about 20-26%, and **79-94% of all units are sold
on promotion** (24-packs are the most dependent). Regular-price sales are the exception, which is
why promotion effectiveness and trade-spend ROI matter as much as list price.
"""),
        md("## 4. Price vs. volume -- a first look at elasticity"),
        code('pd.concat([eda.price_volume(d["panel"], "2L"), eda.price_volume(d["panel"], "24pk can")], axis=1)'),
        fig("eda_price_volume_2L.png"),
        fig("eda_price_volume_24pk_can.png"),
        md("""
**So what:** On a log-log scale volume falls almost linearly as price rises, and steeply: the raw
slopes are about -3 to -4 for 2-litre bottles and -5 to -7 for 24-pack cans. Shoppers react
strongly to price -- especially on the large can packs that they can stock up on. These raw slopes
mix price with promotion displays and seasonality, so notebook 02 estimates the elasticity
properly with those effects controlled.
"""),
    ],
    "02_elasticity.ipynb": [
        md("""
# 02 -- Price elasticity (own and cross)

Model (per brand x pack, store-week level, `beverage_demand.elasticity`):

`ln(units) = b_own ln(own price) + sum b_j ln(competitor j price) + promo-type dummies
+ post-promo dummies + holiday dummies + Fourier seasonality + year effects + store fixed effects`

Standard errors are clustered by store. See `docs/methodology.md` for why each piece is there.
"""),
        code(SETUP),
        code('own = pd.read_csv(results("elasticities.csv"))\n'
             'own[["series", "own_elasticity", "ci_low", "ci_high", "n_obs", "n_stores", "r2"]]'),
        fig("elasticity_own.png"),
        code('for t in own.sort_values("own_elasticity")["interpretation"]:\n    print("-", t)'),
        md("""
**So what:** Every core SKU is strongly price-elastic (all own-price elasticities are negative and
well below -1). 24-pack cans (-4.6 to -6.2) react much more than 2-litre bottles (-2.9 to -4.0):
cans are a planned, stock-up purchase, so shoppers time their buying to the price.
"""),
        md("## Cross-price elasticities"),
        code('cross = pd.read_csv(results("cross_elasticities.csv"))\n'
             'cross.pivot_table(index=["pack", "focal_brand"], columns="competitor_brand", values="cross_elasticity").round(2)'),
        fig("elasticity_cross_heatmap.png"),
        code('cross.loc[~cross["identified"], ["series", "competitor_brand", "note"]]'),
        md("""
**So what:** Most identifiable cross-elasticities are positive (brands are substitutes): when Coke
24-packs get more expensive, Pepsi and Diet Pepsi 24-packs gain volume. Some sister brands
(Pepsi / Diet Pepsi, and Coke / Diet Coke in 24-packs) are always priced together, so the data
cannot separate their effects -- the table marks those pairs instead of reporting a misleading
number. A few negative values (e.g. private-label 24-packs vs. Pepsi) are most likely joint
chain-wide promotional events rather than true complementarity; treat them with caution.
"""),
        md("## Promotion effects, post-promotion dip, robustness"),
        code('eff = pd.read_csv(results("promo_effects.csv"))\n'
             'eff.pivot_table(index="series", columns="term", values="coef_pct").round(3)'),
        md("""
Promo-type coefficients are the volume effect *on top of* the price cut (display/feature support).
`post_promo_1/2` measure the dip after a promotion ends (pantry loading).
"""),
        code('pd.read_csv(results("elasticity_robustness.csv")).round(2)'),
        md("""
**So what:** The estimates are stable: a pooled model with shared controls and a model that lets
elasticity differ by store price tier give the same picture (see `results/summary.json` for the
correlation and the largest tier deviation). The main caveat is **endogeneity** -- the retailer
chooses when to cut prices -- discussed in `docs/methodology.md`.
"""),
    ],
    "03_forecasting.ipynb": [
        md("""
# 03 -- Demand forecasting: LightGBM vs. baselines

Target: weekly units at brand x pack x store, 1-8 weeks ahead. Rolling-origin backtest with 9 folds
(origins every 8 weeks, test weeks 328-399). Code: `beverage_demand.forecasting.backtest`.
Planned price and promotion for the target week are treated as known in advance.
"""),
        code(SETUP),
        md("""
## 1. Tuning LightGBM with time-series cross-validation

8 settings from a small grid, each scored on 3 time-based folds that lie *before* the backtest
period (`beverage_demand.forecasting.tuning`). Every trial is stored in the `tuning_trials` table.
"""),
        code('con.sql("SELECT * EXCLUDE (tuned_at) FROM tuning_trials ORDER BY cv_wape").df().round(4)'),
        md("""
Deeper trees (127 leaves) and a Tweedie power of 1.7 work best: weekly volumes are spiky
(huge promotion weeks, many small weeks), and a higher Tweedie power treats large values as
naturally more variable.

## 2. Backtest: two ML models vs. three baselines

Accuracy is computed in SQL from the `forecast_predictions` table
(`sql/analysis/forecast_accuracy.sql`).
"""),
        code('m = pd.read_csv(results("forecast_metrics.csv"))\n'
             'm[m["slice"].isin(["overall", "promo", "chain_level"])].pivot_table(index=["slice", "group"], columns="model", values="wape").round(3)'),
        fig("forecast_wape_promo_split.png"),
        md("""
**So what:** LightGBM has the lowest error overall and in promotion weeks, ahead of the second
tree model (HistGradientBoosting) and the regression. Naive baselines are
useless here because volume swings roughly 5-15x between promoted and normal weeks; the log-log regression
is a solid baseline in normal weeks but loses accuracy in promo weeks, where LightGBM's
interactions between discount depth, promo type and recent sales help most.
"""),
        code('m[m["slice"] == "series"].pivot_table(index="group", columns="model", values="wape").round(3)'),
        fig("forecast_wape_by_series.png"),
        fig("forecast_actual_vs_forecast.png"),
        fig("forecast_wape_by_horizon.png"),
        md("""
## 3. Feature ablation -- which inputs matter?

The tuned LightGBM is re-trained on the same folds with one group of features removed at a time.
"""),
        code('pd.read_csv(results("ablation.csv")).round(4)'),
        fig("forecast_ablation.png"),
        md("""
**So what:** Removing the price & promotion features hurts by far the most -- without knowing the
promotion plan, the forecast cannot see the spikes coming. Sales history is the second most
important group. Dropping the calendar/holiday features does not hurt (it is even marginally
better): last year's same-week sales and the promotion plan already carry that information.
"""),
        md("## 4. What drives the forecast (SHAP)"),
        code('pd.read_csv(results("forecast_shap_importance.csv")).head(12)'),
        fig("forecast_shap_summary.png"),
        md("""
**So what:** Discount depth is by far the biggest driver of next week's volume, followed by the
store's recent sales level and the price itself. In this category a forecast that does not know
the promotion plan cannot be accurate -- which is why sales and demand-planning teams need to share
the promo calendar.
"""),
    ],
    "04_promo.ipynb": [
        md("""
# 04 -- Promotion lift and ROI

Baseline (what would have sold without the promotion) = the backtest LightGBM model's prediction
for the same week with promotion switched off and price set to regular price. All promo weeks
analysed are out-of-sample. Code: `beverage_demand.promo.analysis`.

* lift = actual - baseline units
* discount investment = (regular - promo price) x units sold
* ROI = incremental units x regular unit margin / discount investment (ROI > 1 = margin-accretive)
"""),
        code(SETUP),
        code('pd.read_csv(results("promo_by_type.csv"))[["pack", "promo_label", "promo_weeks", "avg_discount", "lift_pct", "roi", "net_margin_impact"]]'),
        fig("promo_lift_roi_by_type.png"),
        code('pd.read_csv(results("promo_by_depth.csv"))[["pack", "depth_bucket", "promo_weeks", "avg_discount", "lift_pct", "roi", "discount_investment", "net_margin_impact"]]'),
        fig("promo_roi_by_depth.png"),
        md("""
**So what:** Promotions move a lot of volume, but on average they do not pay back the margin
given away. The pattern by depth is clear: **10-20% discounts are margin-accretive (ROI above 1),
while 30%+ discounts lose money** -- and that is where about two-thirds of the discount money is
spent. Price cuts that are not tagged as a merchandised promotion (no bonus-buy / sale code) lift
volume far less than tagged ones and have the worst ROI (about 0.5).
"""),
        md("## Ranking"),
        code('Markdown(results("promo_ranking.md").read_text())'),
        md("## Post-promotion dip and cannibalisation"),
        code('pd.read_csv(results("promo_by_series.csv"))[["series", "avg_discount", "lift_pct", "roi", "avg_competitor_volume_change_pct", "competitors_measured"]]'),
        md("""
**So what:** After a national-brand 24-pack promotion, volume dips by roughly 8-18% the following
week (shoppers stocked up; see `promo_effects.csv`); 2-litre dips are small, and private-label
24-packs show no dip. Most brands' promotions also take volume from competitors in the same pack
-- up to about 17% of competitors' volume for Coke 24-packs -- so part of the lift is brand
switching, not new category demand.
"""),
    ],
    "05_scenarios.ipynb": [
        md("""
# 05 -- Scenario planning

A small what-if tool built on the elasticity models (`beverage_demand.scenario`). Assumptions:
constant elasticity, no competitor price reaction, fixed unit cost -- directional, not a forecast.
"""),
        code(SETUP + "\nfrom beverage_demand.cli import format_scenario\nfrom beverage_demand.scenario.simulate import simulate"),
        md('## "If we drop Coke 2L by 10% for 4 weeks, what happens?"'),
        code('print(format_scenario(simulate("Coke", "2L", -10, weeks=4)))'),
        md("""
**So what:** A 10% cut on Coke 2L lifts its volume by about 45% and revenue by about 30%, but gross
margin rises much less because every unit now earns less. Roughly 2-3% of the volume of Pepsi,
Diet Pepsi, 7Up and private label 2L moves to Coke.
"""),
        code('print(format_scenario(simulate("Pepsi", "24pk can", -15, promo_type="S", weeks=2)))'),
    ],
}


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                        "language": "python"}
        nbf.write(nb, NB_DIR / name)
        print("wrote", NB_DIR / name)


if __name__ == "__main__":
    main()

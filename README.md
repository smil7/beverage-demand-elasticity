# Beverage Demand Forecasting & Promotion Analytics

[![ci](https://github.com/smil7/beverage-demand-elasticity/actions/workflows/ci.yml/badge.svg)](https://github.com/smil7/beverage-demand-elasticity/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue) ![LightGBM](https://img.shields.io/badge/model-LightGBM-green)

Using 17.7 million rows of real supermarket scanner data (soft drinks, 93 Chicago stores,
1989-1997), I built a model that forecasts weekly demand per product and store, measured how
price-sensitive shoppers are, and worked out which promotions actually pay back.

![Model comparison](results/figures/forecast_wape_promo_split.png)

## Questions

1. How accurately can we forecast weekly sales per product and store, 1-8 weeks ahead -- and how
   much better is a machine-learning model than simple rules?
2. How much does volume change when the price changes (price elasticity)?
3. Which promotions bring in more margin than they cost?

## Results

All numbers come from the code and are saved in [`results/summary.json`](results/summary.json).

| | Result |
|---|---|
| Data | 17.7M raw rows -> 10.7M after cleaning -> 2.39M weekly rows (brand x pack x store x week); 93 stores, 390 weeks. Modelled: 12 products (Coke, Diet Coke, Pepsi, Diet Pepsi, 7Up, store brand x 2-litre / 24-pack), 1,026 store-product series |
| **Forecast** (WAPE = error as a share of volume, lower is better) | **LightGBM 0.473** vs. log-log regression 0.580, seasonal naive 0.908, 4-week average 0.976 |
| | **48% lower error than seasonal naive**, 18% lower than the best baseline; in promotion weeks 0.483 vs. 0.766 for seasonal naive |
| Tuning | 8 LightGBM settings x 3 time-based folds; best CV WAPE 0.464 vs. 0.509 for the worst |
| Price elasticity | all 12 products are very price-sensitive: -2.9 to -6.2 (24-packs more than 2-litres) |
| Promotions | promoted weeks sell about 7.5x the no-promotion baseline, but on average return only $0.90 of margin per $1 of discount; **10-20% discounts pay back (ROI 1.31-1.39), 30%+ discounts don't (0.72-0.83)** |

## How I did it

**1. Data** ([`data/load.py`](src/beverage_demand/data/load.py)) -- read the raw CSV with pandas,
apply the cleaning rules from the dataset's codebook (unit price = price / bundle size, drop rows
flagged as bad or with no price), map 1,746 product descriptions to brands and pack sizes with
rules in [`config/brand_mapping.csv`](config/brand_mapping.csv), and add up sales per brand, pack,
store and week.

**2. Features** ([`features/panel.py`](src/beverage_demand/features/panel.py)) -- regular (shelf)
price as the highest price of the last 13 weeks, discount depth, promotion type, competitor
prices, holidays and seasonality.

**3. Forecasting** ([`forecasting/`](src/beverage_demand/forecasting)) -- one LightGBM model for all
products and stores, forecasting 1-8 weeks ahead:
- **No leakage:** sales features (lags, rolling averages) only use weeks before the forecast is
  made; a unit test checks this exactly. The planned price/promotion of the target week is used,
  since promotions are planned weeks ahead.
- **Time-based validation only:** tuning on 3 folds before the test period, then a 9-fold
  rolling backtest on weeks 328-399 (each fold trains on the past, forecasts the next 8 weeks).
- **Baselines:** same week last year, 4-week average, and a log-log regression.
- **Explainability:** SHAP shows discount depth is the biggest driver, then recent sales level and price.

**4. Price elasticity** ([`elasticity/`](src/beverage_demand/elasticity)) -- a log-log regression
per product (the coefficient on log price is the elasticity), with store effects, promotion,
holiday and seasonality controls, and standard errors clustered by store.

**5. Promotions** ([`promo/analysis.py`](src/beverage_demand/promo/analysis.py)) -- for each
promoted week, the forecast model predicts what would have sold *without* the promotion. Lift =
actual - baseline; ROI = extra margin from the lift / money given away in discounts.

Notebooks walk through each step: [`01_eda`](notebooks/01_eda.ipynb),
[`02_elasticity`](notebooks/02_elasticity.ipynb), [`03_forecasting`](notebooks/03_forecasting.ipynb),
[`04_promo`](notebooks/04_promo.ipynb). More detail in [`docs/methodology.md`](docs/methodology.md).

## Main findings

- **Promotions drive this category:** 79-94% of units of the main products sell on some kind of
  deal, and the planned discount is the single most important forecast input.
- **The ML model clearly beats simple rules,** most of all in promotion weeks, where volume jumps
  roughly 5-20x.
- **Moderate discounts pay back, deep ones don't.** 30%+ discounts lose margin yet take two-thirds
  of the discount budget. A short write-up for a non-technical audience:
  [`reports/executive_summary.md`](reports/executive_summary.md).

## How to run

```bash
uv sync                                    # install dependencies
uv run python scripts/download_data.py     # raw files -> data/raw (see data/README.md)
uv run python scripts/run_all.py           # whole analysis
uv run pytest                              # tests
```

**Data licence:** Dominick's data are provided by the James M. Kilts Center for Marketing,
University of Chicago Booth School of Business, for academic research. The data are not included
in this repository; `results/` holds only aggregated outputs.

## Limitations

- One retailer and 1990s data; the method transfers, the numbers would need re-estimating.
- The retailer chooses when to discount, so the elasticities are probably on the high side.
- Promotion ROI uses the retailer's margin only; extra store traffic from deals is not counted.

## Tech stack

Python, pandas, LightGBM, statsmodels, SHAP, matplotlib, pytest

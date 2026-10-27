# How I built it

A walk-through of the main decisions. Numbers are from `results/summary.json`.

---

## 1. The database

**Data.** Dominick's Finer Foods scanner data, soft drinks: 17.7M rows of store x product (UPC) x
week, 93 Chicago stores, Sep 1989 - May 1997 (Kilts Center, Chicago Booth).

**Why a database.** I wanted every step from raw file to model input to be reproducible SQL that I
can test, not a chain of notebook cells. DuckDB is an analytical database that runs inside Python:
it loads the 1.2 GB CSV and builds the fact table in about 10 seconds, and the same SQL runs on a
tiny in-memory database in the unit tests.

**Layers** (`sql/warehouse/`):

1. `stg_movement` -- the raw CSV with explicit column types.
2. `movement_clean` (view) -- the codebook rules: `PRICE` is the price of a bundle of `QTY` items,
   so unit price = `PRICE / QTY` and dollars = `PRICE * MOVE / QTY`; rows with `OK = 0` (flagged as
   suspect) or `PRICE <= 0` (not on sale that week) are dropped; cost = dollars x (1 - margin%).
3. Star schema -- `fact_sales_weekly` (brand x pack x store x week, composite **primary key**) plus
   `dim_product`, `dim_store` and `dim_calendar`, each with a primary key. Price at the weekly
   grain is the volume-weighted average (total dollars / total units).
4. `etl_run_log` -- row counts of every load.

**Product mapping.** 1,746 UPC descriptions are mapped to brands with ordered regular expressions in
`config/brand_mapping.csv` (first match wins, so "Diet Coke" is tested before "Coke"). Pack size is
parsed from the size text (`24/12O` = 24 cans of 12 oz). Unit tests pin the tricky cases.

**Data-quality checks** (`sql/quality/checks.sql`) run after every load: duplicate keys, negative
volumes, non-positive prices, orphan UPCs / weeks / stores, implausible margins, missing weeks.
They found 9 weeks with no data anywhere in the chain, 7 stores without a demographics row (given
price tier "Unknown"), and 1.2% of rows with impossible margins (those costs are ignored).

## 2. Features in SQL

`sql/features/model_panel.sql` builds the table every model uses, for 12 products (Coke, Diet Coke,
Pepsi, Diet Pepsi, 7Up and the store brand, in 2-litre bottles and 24-pack cans) -- 381,397 rows:

| Step | What | SQL used |
|---|---|---|
| Grid | one row per store x product x week between first and last sale; weeks with no data in the whole chain are marked as gaps | join to `dim_calendar` on a week range |
| Price | carry the last known price forward when an item did not sell | `LAST_VALUE(... IGNORE NULLS) OVER (...)` |
| Regular price | highest price in the last 13 weeks (current week included, never future weeks) | `MAX() OVER (ROWS BETWEEN 12 PRECEDING AND CURRENT ROW)` |
| Discount, promo type | discount = 1 - price / regular price; promo type from the sale code, or "unflagged" if the price is >= 5% below regular | `CASE` |
| Post-promotion weeks | the 1st / 2nd week after a promotion ends | `LAG()` |
| Competitor prices | each brand's log price in the same store, pack and week | long table + `PIVOT` |

**Checking the SQL.** I first wrote the same logic in pandas and compared the two outputs: all
381,397 rows matched to within 1e-15. The pandas version was then removed, and unit tests now run
the SQL on small hand-made databases with known answers.

## 3. Forecasting

**Target:** weekly units per product per store, 1 to 8 weeks ahead.

**Features** (`forecasting/features.py`): sales lags (1, 2, 4, 8 and 52 weeks), 4- and 13-week rolling
mean/std, price, regular price, discount, promotion type, competitor prices, holidays,
seasonality, store demographics, and the horizon. One global model is trained on all products,
stores and horizons together.

**Leakage.** For a forecast `h` weeks ahead, every sales feature uses data up to `h` weeks before the
target week only. A unit test builds a series where units = week number and checks each lag
exactly. The price and promotion of the target week *are* used, because promotions are planned
weeks in advance -- the forecast answers "what will we sell given the promotion plan".

**Validation: time-based only.**

```
tuning folds   (origins 303, 311, 319)    -> choose hyperparameters
backtest folds (origins 327, 335 ... 391) -> report accuracy on weeks 328-399
```

Each fold trains on everything up to the origin week and forecasts the next 8 weeks. Random splits
would let the model learn from the future.

**Tuning** (`forecasting/tuning.py`): 8 settings sampled from a small grid (Tweedie power, number of
leaves, minimum rows per leaf), each scored on the 3 tuning folds with early stopping. All trials
are logged to the `tuning_trials` table.

**Models compared** (`forecasting/backtest.py`):

| Model | Why it is there |
|---|---|
| LightGBM (tuned, Tweedie loss) | main model |
| HistGradientBoosting (scikit-learn, Poisson loss) | a second tree model, to check the result is not specific to one library |
| Log-log regression | the elasticity model from section 5, re-fitted on each fold |
| Seasonal naive (same week last year) | simple benchmark |
| 4-week moving average | what a spreadsheet would do |

Both tree models use **monotone constraints** (volume cannot rise when our price rises), so the
model's price response is sensible enough to answer "what if there had been no promotion?".

**Metrics** are computed in SQL (`sql/analysis/forecast_accuracy.sql`) from the
`forecast_predictions` table. The main metric is **WAPE** = total absolute error / total actual
units: big weeks count more, and it does not blow up on tiny weeks like MAPE does. I also report
MAPE, bias and RMSE, split by product, horizon and promo vs non-promo weeks.

**Feature ablation** (`forecasting/ablation.py`): the tuned model is re-trained without one feature
group at a time (sales history, price & promotion, calendar, store demographics) on the same folds,
to see how much each group adds.

**SHAP** values explain the last fold's model.

## 4. Promotion lift and ROI

For each promoted store-week in the backtest period, the LightGBM fold model (trained only on
earlier weeks) predicts the week twice: as it happened, and with the promotion removed (regular
price, no discount, no promo flags). The second number is the **baseline** -- what would have sold
without the promotion.

`sql/analysis/promo_events.sql` then computes, per promoted week:

```
lift                = actual units - baseline units
discount investment = (regular price - promo price) x units sold
incremental margin  = lift x (regular price - unit cost)
ROI                 = incremental margin / discount investment   (ROI > 1 = the promotion added margin)
```

and groups consecutive promoted weeks into events with a gaps-and-islands query.
`promo_roi.sql` aggregates by product, promotion type and discount depth. ROI is only ranked for
groups with at least 50 promoted store-weeks and an average discount of at least 5%.

## 5. Price elasticity

`ln(units) = b x ln(price) + competitor prices + promotion dummies + holidays + seasonality + year
and store effects`, one regression per product (statsmodels OLS).

- In a log-log model `b` is the elasticity: `b = -3.5` means a 1% price increase lowers volume by
  about 3.5%.
- **Store fixed effects** compare each store with itself over time (big stores sell more anyway).
- **Clustered standard errors by store**: weeks of the same store are not independent, so normal
  standard errors would be too small.
- **A problem I found:** some sister brands are always priced together (Pepsi and Diet Pepsi 2L
  prices correlate at 0.99), so the regression cannot separate their effects -- the first run gave
  nonsense numbers. When two prices correlate above 0.9, only one is kept and the pair is reported
  as "not identified" (15 of 60 pairs).
- **Robustness:** a pooled model and a model by store price tier give a consistent picture.
- **Caveat:** the retailer chooses when to cut prices, and almost all price changes are short
  promotions, so the elasticities are probably larger than the response to a permanent price
  change.

## 6. Limitations

- One retailer, 1990s data: the methods transfer, the numbers would need re-estimating.
- The margin view is the retailer's; store traffic and basket effects of promotions are not included.
- Scenario results assume constant elasticities and no competitor reaction.
- Zero-sales weeks are dropped from the log regressions (log of zero).
- Promotion codes in the raw data are incomplete (the provider warns about this); the 5% price rule
  adds its own noise.

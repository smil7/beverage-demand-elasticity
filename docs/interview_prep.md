# Interview prep -- questions about this project

Short answers in my own words. Numbers come from `results/summary.json`.

---

## The project

**1. In one minute, what did you build?**
I loaded 17.7M rows of weekly supermarket scanner data (soft drinks, 93 stores, 1989-1997) into a
DuckDB warehouse with SQL: staging, cleaning, a star schema with primary keys, and the whole
feature table built with window functions. On top of it I trained and compared forecasting models
(tuned LightGBM, scikit-learn HistGradientBoosting, a log-log regression and two naive baselines)
with a rolling time-based backtest, ran a feature ablation, estimated price elasticities, and used
the forecast model to measure which promotions pay back. Results and predictions go back into the
database and are analysed with SQL.

**2. Why did you pick this dataset?**
It is real retail data with prices, promotions and margins -- the same problems a beverage company
works on (demand, pricing, trade promotions) -- and it is big enough (17.7M rows) that the data
engineering part is real.

## Databases and SQL

**3. Why DuckDB and not pandas or PostgreSQL?**
The work is analytical (big scans, group-bys, window functions), which is what DuckDB is built for:
it loads the 1.2 GB CSV and builds the fact table in about 10 seconds on a laptop, needs no server,
and the same SQL file runs in CI tests on an in-memory database. The SQL is standard enough
(CTEs, window functions, `PRIMARY KEY`, `NOT NULL`) to move to PostgreSQL or Snowflake with small
changes.

**4. Walk me through the data model.**
Staging table `stg_movement` (raw, typed) -> view `movement_clean` (codebook rules: unit price =
PRICE/QTY, drop OK = 0 and PRICE <= 0) -> star schema: `fact_sales_weekly` at brand x pack x store x
week with a composite primary key, and dimensions `dim_product` (UPC -> brand, pack),
`dim_store` (price tier, demographics) and `dim_calendar` (dates, holidays). Then `model_panel`
for modelling and output tables like `forecast_predictions` and `promo_events`.

**5. How did you build features in SQL?**
`sql/features/model_panel.sql`, step by step in temp tables:
- a complete weekly grid per store x series (join to the calendar between first and last sale),
- forward-fill missing prices with `LAST_VALUE(... IGNORE NULLS) OVER (... ROWS BETWEEN UNBOUNDED
  PRECEDING AND CURRENT ROW)`,
- regular price as a rolling 13-week `MAX() OVER (ROWS BETWEEN 12 PRECEDING AND CURRENT ROW)`,
- promotion type with `CASE`, post-promotion weeks with `LAG()`,
- competitor prices with a long table + `PIVOT`.

**6. How do you know the SQL features are correct?**
Two ways. I first wrote the same logic in pandas and compared both outputs row by row: all 381,397
rows matched (differences below 1e-15). Then I wrote unit tests that run the SQL on tiny in-memory
databases where I know the answer (e.g. a price rise in week 4 must not change the regular price
of weeks 1-3).

**7. What is "gaps and islands" and where did you use it?**
Grouping consecutive rows into runs. A promotion "event" is a run of consecutive promoted weeks:
`week - ROW_NUMBER() OVER (PARTITION BY series, store ORDER BY week)` is constant inside a run, so
it identifies the event (`sql/analysis/promo_events.sql`).

**8. What data-quality checks did you run?**
`sql/quality/checks.sql`: no duplicate keys (also enforced by the primary key), no negative volumes,
positive prices, every UPC/week/store found in its dimension, margins in a plausible range, week
coverage. Blocking checks stop the pipeline; others are warnings. They found real issues: 9 weeks
missing chain-wide, 7 stores with no demographics row, and 1.2% of rows with impossible margins.

**9. How would this change in production?**
Same layers in a cloud warehouse (Snowflake / Databricks): scheduled loads, the SQL as versioned
models, data-quality checks as tests that block bad loads, the training job reading `model_panel`
and writing predictions back to a table. I already log every tuning trial and backtest run to
tables (`tuning_trials`, `experiment_runs`), which is a small version of experiment tracking.

## Modelling and training

**10. How did you avoid data leakage?**
- Time-based validation only, never random splits.
- Every sales-based feature for a target week uses data up to the forecast origin only (lags,
  rolling means). A unit test builds a series where units = week number and checks every lag for
  horizons 1, 3 and 8.
- The regular price uses only past and current weeks (also tested).
- Hyperparameters were tuned on folds *before* the backtest period.

**11. Explain your validation setup.**
Rolling origin: train on everything up to week T, forecast weeks T+1 to T+8, move T forward 8 weeks,
repeat 9 times (test weeks 328-399). Tuning used 3 earlier folds (origins 303, 311, 319).

**12. How did you tune LightGBM?**
A small grid (Tweedie power, number of leaves, minimum rows per leaf), 8 sampled settings, each
scored by WAPE on the 3 tuning folds with early stopping choosing the number of trees. The best
setting (Tweedie 1.7, 127 leaves) had CV WAPE 0.469 vs 0.516 for the worst. All trials are in
`results/tuning_trials.csv`.

**13. Why Tweedie loss?**
Weekly units are non-negative and very skewed: many ordinary weeks and some huge promotion weeks.
Tweedie (and Poisson) losses are made for that; plain squared error would be dominated by the
biggest weeks. CV confirmed that a higher Tweedie power was better.

**14. Why compare two tree models?**
To show the result is not a LightGBM accident. HistGradientBoosting is a different implementation
of the same idea; both clearly beat the baselines (WAPE 0.475 and 0.499 vs 0.580 for the best
baseline), so most of the gain comes from the features and the approach. LightGBM won overall
but not in every fold (HistGradientBoosting was better in 2 of 9), and LightGBM was tuned while
HistGradientBoosting used fixed settings -- so the gap between them is small and partly tuning.

**15. What did the feature ablation tell you?**
Removing the price and promotion features hurts by far the most (WAPE 0.475 -> 0.843): the model
cannot predict promotion spikes without the promotion plan. Sales history comes next (+0.016),
then store demographics (+0.007). Removing the calendar/holiday features made the model very
slightly *better* (-0.002) -- the same-week-last-year lag and the promotion features already carry
that information, so those features are candidates to drop. See `results/ablation.csv`.

**16. Isn't using the price of the forecast week leakage?**
No, it is planned in advance: retailers and suppliers agree prices and promotions weeks ahead.
I document it as an assumption -- the forecast is "volume given the promotion plan".

**17. Why WAPE instead of MAPE?**
WAPE = total absolute error / total actual volume, so big weeks count more -- which matters for
supply. MAPE explodes on tiny weeks (forecast 5, actual 1 = 400% error). I report both.

**18. Why monotone constraints?**
So the model cannot learn "higher price -> more volume" from noise. That matters because I ask the
model "what if there had been no promotion?" to measure promotion lift.

**19. What is SHAP and what did it show?**
It splits each prediction into feature contributions. Discount depth is the top driver, then the
recent 13-week sales level and the price.

## Business results

**20. What is a price elasticity, and what did you find?**
The % change in volume for a 1% price change. I estimated it with a log-log regression (coefficient
on log price = elasticity) with store fixed effects and store-clustered standard errors. All 12
products are very price-sensitive (-2.9 to -6.2); 24-packs more than 2-litre bottles. One caveat:
the retailer chooses when to discount, so these are probably upper bounds.

**21. Which promotions pay back?**
Moderate discounts (10-20% off) earn back more margin than they cost (ROI 1.30-1.32); deep 30%+
discounts do not (0.72-0.81), and they take two-thirds of the discount budget.

**22. What would you do next?**
Hierarchical forecasts (store -> chain), better handling of the price-endogeneity problem, and
moving the pipeline to a scheduled cloud job with monitoring of forecast error.

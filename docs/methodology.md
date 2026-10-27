# How I built it

The main decisions, in order. Numbers are from `results/summary.json`.

## 1. Data

Dominick's Finer Foods scanner data, soft drinks: 17.7M rows of store x product (UPC) x week from
93 Chicago stores, Sep 1989 - May 1997 (Kilts Center, Chicago Booth).

Cleaning (`src/beverage_demand/data/load.py`), following the dataset's codebook:
- `PRICE` is the price of a bundle of `QTY` items, so unit price = `PRICE / QTY` and
  dollar sales = `PRICE * MOVE / QTY`.
- Rows flagged as suspect (`OK = 0`) or with no price (item not on sale that week) are dropped:
  17.7M -> 10.7M rows.
- Cost = dollars x (1 - margin %). 1.2% of rows have impossible margins; their cost is ignored.
- 1,746 product descriptions are mapped to brands with ordered rules in
  `config/brand_mapping.csv` ("Diet Coke" is checked before "Coke"), and the pack size is read
  from the size text (`24/12O` = 24 cans of 12 oz).
- Sales are added up to brand x pack x store x week (2.39M rows). The weekly price is the
  volume-weighted average (dollars / units).

## 2. Features

`src/beverage_demand/features/panel.py`, for 12 products (6 brands x 2-litre / 24-pack):
- weeks where a product did not sell get 0 units; 9 weeks with no data in the whole chain are
  marked as gaps and skipped,
- **regular price** = highest price of the last 13 weeks (current week included, never future
  weeks), **discount** = 1 - price / regular price,
- **promotion type** from the data's sale code, or "unflagged" if the price is >= 5% below regular,
- competitor prices in the same store and pack, holidays (incl. Super Bowl), seasonality, store
  demographics.

## 3. Forecasting

**Target:** weekly units per product per store, 1-8 weeks ahead. One LightGBM model covers all
products, stores and horizons.

**Features** (`forecasting/features.py`): sales lags (1, 2, 4, 8, 52 weeks), 4- and 13-week rolling
mean/std, price, discount, promotion type, competitor prices, calendar and store features.

**No leakage.** For a forecast `h` weeks ahead, every sales feature uses only weeks at least `h`
weeks before the target week; a unit test builds a series where units = week number and checks
each lag exactly. The target week's planned price and promotion *are* used -- promotions are
planned weeks ahead, so the forecast answers "what will we sell given the promotion plan".

**Validation -- time-based only:**

```
tuning folds   (origins 303, 311, 319)    -> choose LightGBM settings
backtest folds (origins 327, 335 ... 391) -> report accuracy on weeks 328-399
```

Each fold trains on everything up to its origin week and forecasts the next 8 weeks. A random split
would let the model learn from the future.

**Tuning** (`forecasting/tuning.py`): 8 settings from a small grid (Tweedie power, number of
leaves, minimum rows per leaf), each trained for up to 300 trees on the 3 tuning folds. The
validation error curves of the folds are averaged and the number of trees with the lowest average
error is used. Best: Tweedie power 1.7, 127 leaves, 50 rows per leaf, 300 trees -- CV WAPE 0.464
(worst 0.509). The best number of trees hit the 300 limit, so a few more might help slightly. Tweedie loss suits sales data: never
negative, many ordinary weeks and a few huge promotion weeks.

**Monotone constraints:** the model may not predict more volume when our price goes up. This
keeps the price effect sensible, which matters for the promotion analysis below.

**Baselines:** same week last year, 4-week average, and the log-log regression from section 4.

**Metric:** WAPE = total absolute error / total actual units. Large weeks count more, and it does not
blow up on tiny weeks like MAPE does (I report MAPE, bias and RMSE too), split by product, horizon
and promotion vs. normal weeks.

## 4. Price elasticity

`ln(units) = b x ln(price) + competitor prices + promotion dummies + holidays + seasonality +
year and store effects`, one regression per product (statsmodels).

- `b` is the elasticity: `b = -3.5` means a 1% price increase lowers volume by about 3.5%.
- Store effects compare each store with itself over time.
- Standard errors are clustered by store, because weeks of the same store are not independent.
- Some sister brands are always priced together (Pepsi and Diet Pepsi 2L prices correlate at 0.99),
  so their separate effects cannot be estimated. When two prices correlate above 0.9, one is
  dropped and the pair is reported as "not identified".
- Caveat: the retailer chooses when to discount, and most price changes are short deals, so these
  elasticities are probably larger than the reaction to a permanent price change.

## 5. Promotions

For each promoted store-week in the test period, the LightGBM model of that fold (trained only on
earlier weeks) predicts the week twice: as it happened, and with the promotion removed (regular
price, no discount). The second number is the baseline.

```
lift                = actual units - baseline units
discount investment = (regular price - promo price) x units sold
incremental margin  = lift x (regular price - unit cost)
ROI                 = incremental margin / discount investment   (> 1 means it paid back)
```

Results are grouped by product, promotion type and discount depth; only groups with at least 50
promoted store-weeks and an average discount of at least 5% are ranked.

## 6. Limitations

- One retailer, 1990s data.
- Margin is the retailer's; extra store traffic or basket spend from deals is not counted.
- Zero-sales weeks are left out of the log regressions (log of zero).
- The raw promotion codes are incomplete (the data provider says so); the 5% price rule adds noise.

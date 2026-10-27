# Methodology

Written for a smart reader who has not done econometrics before. Every number quoted here is in
`results/summary.json` or the CSVs in `results/`.

---

## 1. Data and the modelling grain

**Source.** Dominick's Finer Foods weekly scanner data, soft-drink category: 17.7M rows of
store x UPC x week, 93 stores, Sep 1989 - May 1997 (Kilts Center, Chicago Booth).

**Cleaning** (codebook rules, `sql/01_movement_clean.sql`): drop `OK = 0` (flagged by the data
provider as suspect), `PRICE <= 0` (item not on sale that week -- almost all of the 6.9M dropped
rows) and `MOVE < 0`. `PRICE` is the price of a bundle of `QTY` items, so unit price is
`PRICE / QTY` and dollar sales are `PRICE x MOVE / QTY`.

**Product hierarchy.** 1,746 UPCs are mapped to brand with ordered regular expressions in
`config/brand_mapping.csv` (first match wins, so "Diet Coke" is tested before "Coke"; flavour
extensions such as Cherry Coke or Wild Cherry Pepsi are kept out of the core brands). Pack size is
parsed from the codebook `SIZE` field (`24/12O` -> 24-pack of 12-oz cans; `2 LT` -> 2-litre).
Unit tests pin the tricky cases.

**Grain.** `fact_sales_weekly` = brand x pack x store x week. Price at this grain is the
**volume-weighted average unit price** (total dollars / total units), so a week where most volume
sold on the promoted UPC gets the promoted price. The chain-level `fact_brand_week` adds packs
together in **8-oz equivalent cases** (192 oz), the standard soft-drink volume unit.

**Scope.** Modelling focuses on 12 series: Coke, Diet Coke, Pepsi, Diet Pepsi, 7Up and Dominick's
private label, each in 2-litre bottles and 24-pack cans -- the biggest national brands plus the
retailer's own brand, in the two most important packs. 1,026 store x series combinations with at
least 100 selling weeks are modelled.

## 2. Promotions and the "regular price"

Retail prices have two parts: a **regular (shelf) price** that changes rarely, and **temporary price
reductions**. We need to separate them.

* `reg_price` = the highest price in the trailing 13 weeks *including the current week*. Using only
  past and current weeks means a later price rise cannot leak backwards (a unit test checks this).
* `discount` = `1 - price / reg_price`.
* A week is a **promotion** if the data's `SALE` code is set (B = bonus buy, S = simple price
  reduction, C = coupon; `G` appears but is undocumented) **or** the price is at least 5% below
  regular. The codebook explicitly warns that `SALE` is not recorded consistently, which is why the
  price rule is needed; those weeks are labelled `U` (unflagged price cut).

A known weakness: a permanent price drop looks like a "discount" for up to 13 weeks until the
regular price catches up.

## 3. Price elasticity

### Why log-log?

We fit

```
ln(units) = b_own * ln(own price) + sum_j b_j * ln(price of competitor j) + controls + error
```

In a log-log model the coefficient on log price **is** the elasticity: the % change in volume for a
1% change in price. `b_own = -3.5` means a 1% price rise cuts volume by about 3.5% (and a 10% cut
raises it by `0.9^-3.5 - 1 = 45%`). Logs also make the multiplicative structure of retail demand
(a promotion multiplies sales, it doesn't add a fixed number of cases) additive, and they tame the
very skewed sales distribution.

**How to read an elasticity.** Below -1 ("elastic"): a price cut raises *revenue*. Whether it raises
*margin* depends on the margin too (section 6). Cross-elasticity `b_j > 0`: when competitor j gets
more expensive we sell more -- the brands are **substitutes**.

### Controls, and why each is there

| Control | Why |
|---|---|
| Promo-type dummies (B, S, C, G, U) | Promotions come with displays and feature ads. Without these, the display effect would be credited to price and the elasticity would be too large. With them, `b_own` is the pure price response and the dummy is the extra "merchandising" lift. |
| Post-promotion dummies (1 and 2 weeks after) | Shoppers stock up on deal and buy less afterwards ("pantry loading"). |
| Holiday dummies (10 holidays incl. Super Bowl) | Holiday weeks are both high-demand and heavily promoted; ignoring them would make promotions look more effective. |
| Fourier terms (3 sine/cosine pairs on day-of-year) | Smooth summer/winter seasonality with only 6 parameters. |
| Year fixed effects | Category trends, inflation, store openings. |
| Store fixed effects | Big stores sell more *and* may price differently; fixed effects compare each store only with itself over time. |

### Clustered standard errors

The 30,000 store-weeks of a series are not 30,000 independent observations: a store that sells
more than predicted this week usually does next week too. Ordinary standard errors would assume
independence and be far too small. **Clustering by store** allows any correlation within a store
over time and only treats different stores as independent. The resulting confidence intervals are
honest about how much information there really is.

### A data problem we hit: lockstep pricing

In this data some sister brands are always priced together -- within a store, the log price of
Pepsi and Diet Pepsi 2L correlates at 0.99, and Coke and Diet Coke 24-packs at 0.91-0.93. A
regression cannot tell apart two variables that always move together; the first version produced
nonsense (Diet Pepsi 2L own-elasticity -0.75, Pepsi cross-elasticity -3.2). The fix
(`elasticity/model.py::make_spec`): when a competitor's price correlates > 0.9 with the own price,
it is dropped (its effect is absorbed into the own-price coefficient, which then means "the brand
family moving price together"); when two competitors move together, only one is kept and the
other is reported as **not identified**. 15 of 60 cross pairs are flagged this way rather than
reported with a misleading number.

### Sanity checks and robustness

* All 12 own-price elasticities are negative (-2.9 to -6.2), with tight CIs.
* 80% of identifiable cross-elasticities are positive (substitutes), 69% significantly so. The
  negative ones (e.g. private-label 24-packs vs. Pepsi price, -0.54) are most plausibly chain-wide
  promotional events that hit several brands at once rather than real complementarity.
* **Pooled model** (one regression for all 12 series, own price interacted with series, shared
  controls, no competitor prices): correlation 0.97 with the per-series estimates. It agrees on
  24-packs and gives smaller 2L elasticities (-2.3 to -3.0 vs -2.9 to -4.0) -- without competitor
  prices, the volume lost when rivals discount at the same time is attributed to our own price.
* **Price-tier interactions** (elasticity allowed to differ for Low / Medium / High price-tier
  stores): no tier deviates from the baseline by more than 0.4.

### Endogeneity -- the honest caveat

The textbook assumption behind a regression is that price is "as good as randomly assigned". It
is not: the retailer and manufacturers **choose** when to cut prices, and they do it in weeks they
expect demand to be high (summer holidays) or when a manufacturer pays for it. If price cuts land
in high-demand weeks not captured by our controls, the model credits the price cut with demand
that would have happened anyway, and the elasticity looks bigger than it is. The holiday,
seasonality and promotion controls remove the obvious versions of this, but not all of it.

A second, related bias: almost all price variation here is **temporary**. Shoppers respond to a
one-week deal by buying for several weeks (stock-piling), so short-run elasticities overstate how
volume would react to a **permanent** price change. Both biases push in the same direction, so the
elasticities should be read as upper bounds on long-run price sensitivity.

Ways to do better (not done here): instrument price with wholesale cost (`PROFIT` gives cost, which
moves price for supply-side reasons), use prices in other cities for the same UPC (Hausman
instruments), or run a pricing experiment.

## 4. Forecasting

### Setup

* **Target:** weekly units, brand x pack x store; **horizon** 1-8 weeks.
* **Direct multi-horizon strategy:** one global LightGBM is trained on all horizons at once, with
  `h` as a feature. For a target week `t` forecast `h` weeks ahead, every sales-based feature uses
  data up to `t - h` only (`lag_k = units at t-(h+k-1)`, rolling means/std ending at `t-h`, plus
  same-week-last-year). Unit tests construct a series where units = week number and assert this
  exactly for several horizons.
* **Known-in-advance features:** the planned price, discount and promotion type for the target
  week, competitor price index, holidays, seasonality, store demographics. Retailers set prices
  and promotions weeks ahead and suppliers know the promotion calendar, so using them is
  legitimate -- but it means the accuracy reported is for "forecast given the promo plan", which
  is how demand planning uses it.
* **Monotone constraints:** LightGBM is constrained so volume cannot rise with own price or fall
  with discount depth. This costs little accuracy and makes the model's price response sensible
  enough to use for the promotion counterfactual.
* **Objective:** Tweedie loss (made for non-negative, spiky, right-skewed volumes). The variance
  power (1.7) was chosen by `scripts/select_lightgbm_objective.py` on weeks 320-327 -- strictly
  *before* the first backtest origin -- so model selection never looks at the test folds.

### Validation: rolling origin, never random

Random train/test splits leak the future into training: the model would learn from week 380 to
predict week 379. Instead, the **rolling-origin backtest** uses 9 forecast origins (weeks 327, 335,
..., 391). For each origin `T` the model is trained on weeks `<= T` only and forecasts weeks
`T+1 ... T+8`. Test weeks 328-399 (about 1.4 years) are each forecast exactly once, at their true
horizon.

### Baselines

| Baseline | Why it is there |
|---|---|
| Seasonal naive (same week last year) | The classic "do nothing clever" benchmark |
| 4-week moving average | What many planning spreadsheets do |
| Log-log regression from section 3 | Re-fitted on each fold's training window (with a trend instead of year dummies so it can extrapolate) -- tests whether ML adds anything over the econometric model |

### Metrics -- why WAPE first

**WAPE** = sum of absolute errors / sum of actual units. It is a volume-weighted error: missing
by 50 cases on a 500-case week counts 10x more than missing by 5 on a 50-case week -- which is
how the supply chain experiences errors. **MAPE** averages % errors per row and explodes on
low-volume weeks (5 units forecast for an actual of 1 is a 400% error), so on intermittent,
promotion-driven store-level series it mostly measures noise on tiny rows. MAPE is reported (on
rows with positive actuals) because people ask for it, along with **bias** (are we systematically
over/under?) and RMSE. Everything is split by brand, horizon, and **promo vs non-promo weeks**.

### Results (backtest, store level)

| | LightGBM | Regression | Seasonal naive | 4-wk MA |
|---|---|---|---|---|
| WAPE overall | **0.475** | 0.580 | 0.908 | 0.976 |
| WAPE promo weeks | **0.485** | 0.598 | 0.766 | 0.742 |
| WAPE non-promo weeks | **0.400** | 0.449 | 1.975 | 2.737 |

Naive baselines fail because volume moves 4-16x between promoted and normal weeks: last year's
same week was promoted when this week is not (or vice versa). The regression is close to LightGBM
in normal weeks (0.449 vs 0.400) because a constant-elasticity model describes regular weeks well;
LightGBM's advantage is biggest in **promo weeks**, where the volume response depends on
interactions (depth x promo type x pack x recent sales level) that a single elasticity cannot
capture. LightGBM under-forecasts by about 9% (bias -0.088), mainly on the largest promotion spikes.

Store-level weekly WAPE of ~0.48 sounds high, but the level matters: these are single-store,
single-pack weekly volumes in a category where 79-94% of volume is sold on deal. Summed over
stores to brand x pack x week (what a brand planner looks at), WAPE improves only moderately, to
0.426 (seasonal naive 0.881, regression 0.519; `chain_level` rows of
`results/forecast_metrics.csv`). Store errors do not cancel out much because promotions run
chain-wide: when the model misjudges a deal, it misjudges it in every store at once.

**SHAP** (Shapley values) splits each prediction into contributions per feature. Discount depth is
the top driver, then the 13-week sales level, own price and brand x pack.

## 5. Promotion lift, trade spend and ROI

### Baseline (counterfactual) volume

To know what a promotion *added*, we need what would have sold **without** it. For every promo
store-week in the backtest period, the fold's LightGBM model (trained only on earlier weeks)
predicts the same week twice: once as it happened, and once with promotion flags off, discount 0
and price set back to regular price (`forecasting.backtest.counterfactual`). The second prediction
is the baseline. Sanity check: the average baseline is close to actual sales in non-promoted weeks
of the same series (e.g. Coke 2L: 111 vs 151 units per store-week; Pepsi 24-pack: 32 vs 28).

### Economics

```
lift_units          = actual - baseline
discount_investment = (regular price - promo price) x actual units
incremental_gm      = lift_units x (regular price - unit cost)
ROI                 = incremental_gm / discount_investment
net margin impact   = incremental_gm - discount_investment
```

The last line is algebraically equal to *actual gross margin - baseline gross margin at regular
price* (unit-tested), so **ROI > 1 exactly when the promotion increased gross-margin dollars**.
Unit cost comes from the `PROFIT` field (retailer gross margin, average acquisition cost), so it
already reflects manufacturer deal money that lowers the retailer's cost during promotions.

ROI is ranked only for promotion groups with at least 50 store-weeks and an average discount of at
least 5% -- below that, the discount cost is close to zero and the ratio explodes meaninglessly.

### Post-promotion dip and cannibalisation

* The `post_promo_1/2` coefficients of the elasticity model measure the dip: about 8-18% the week
  after a national-brand 24-pack promotion (median -13.8% across 24-pack series), ~1% for 2-litre.
* Cannibalisation uses the cross-elasticities: a promotion of depth `d` on brand k changes
  competitor j's volume by `(1 - d)^{b_jk} - 1`.

### What is *not* in the ROI

Traffic and basket effects (a cheap 24-pack brings shoppers who buy other things), manufacturer
trade funds beyond what shows up in acquisition cost, and long-term brand effects. Retailers
rationally run some "loss-making" soft-drink promotions as traffic builders; the ROI here is the
category-margin view only.

## 6. Scenarios and optimisation

`scenario.simulate` applies the elasticities to a baseline (average non-promoted weekly volume over
the last 26 weeks): volume x `(1 + price change)^{b_own}`, times `exp(b_promo)` if a promotion type
is added; competitors in the same pack change by their identified cross-elasticity. Uncertainty
comes from 2,000 draws of each coefficient from N(estimate, SE^2) (independent draws -- covariance
between coefficients is ignored, a simplification).

`scenario.optimize` maximises `(p - c) x Q0 x (p / p0)^e` over `p` within +/-20% of the current
regular price using `scipy.optimize.minimize_scalar`. With constant elasticity the answer is the
Lerner rule `p* = c x e / (1 + e)`, which the optimiser reproduces (Coke 2L: e = -3.49, c = $0.98
-> p* = $1.37). **Assumptions:** constant elasticity, competitors do not react, unit cost fixed --
and, from section 3, an elasticity measured on temporary promotions overstates the response to a
permanent price change. The output is a directional tool for ranking options, not a shelf price.

## 7. Limitations

* **One retailer, 1989-1997.** Prices, pack architecture and promotion practice have changed; the
  methods transfer, the numbers do not.
* **Retailer data, not bottler data.** A bottler sees shipments and trade-spend contracts, not
  shelf scans; margins here are the retailer's.
* **Endogenous prices** and temporary-price-cut identification (section 3).
* **No competitor reaction** in scenarios and optimisation.
* **Zero-sales weeks** are dropped from the log models (log of zero) -- fine for these high-volume
  series, a problem for slow movers.
* **Promotion codes are noisy** (codebook warning); the 5% price rule adds its own noise.
* **Databricks notebooks** were not executed on a Databricks workspace (see `databricks/README.md`).

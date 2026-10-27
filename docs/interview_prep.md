# Interview prep -- questions about this project

Short model answers. Numbers refer to `results/summary.json`.

---

**1. In one minute, what did you build?**
An end-to-end commercial analytics project on 7.5 years of store-level soft-drink scanner data
(17.7M rows, 93 stores). PySpark + SQL build a star schema; log-log regressions estimate own- and
cross-price elasticities for 12 brand-pack SKUs; a global LightGBM forecasts store-level weekly
demand 1-8 weeks ahead and beats seasonal-naive by 48% WAPE (37% in promo weeks); the forecast
model's counterfactual gives promotion lift and ROI; and a scenario tool / optimiser turns
elasticities into "what if we cut price 10%" answers. Outputs feed a Power BI star schema and
Databricks notebooks.

**2. Why a log-log model?**
The coefficient on log price is directly the elasticity, retail demand is multiplicative (a
promotion multiplies sales), and logs tame the skew. It is the industry-standard first model
because it is interpretable and gives a single number a pricing manager can use.

**3. How do you read an elasticity of -3.5?**
A 1% price increase cuts volume by ~3.5%. For a 10% cut: `0.9^-3.5 - 1` = +45% volume. Below -1,
revenue rises when price falls; whether margin rises depends on the margin (Lerner rule).

**4. Why might your elasticity estimates be biased?**
Endogeneity: prices are chosen by the retailer, often in weeks with high expected demand or with
manufacturer funding. If the controls miss part of that demand, the price cut gets credit for it
and the elasticity is too large. Also nearly all price variation is temporary deals, and shoppers
stock up on deals, so short-run elasticities overstate the response to a permanent price change.
Both biases push towards over-estimating sensitivity. Fixes: cost-based or Hausman instruments, or
experiments.

**5. What are clustered standard errors and why use them?**
Weekly errors within a store are correlated (a store that over-sells this week tends to next week).
Ordinary SEs assume independence and would be much too small, making everything look significant.
Clustering by store allows arbitrary correlation within a store over time.

**6. Some cross-elasticities were "not identified". What does that mean?**
Pepsi and Diet Pepsi 2L prices correlate at 0.99 within store -- they are always promoted together.
A regression can't separate two variables that move together; my first run gave Diet Pepsi 2L an
own elasticity of -0.75 and Pepsi a cross-elasticity of -3.2, which is collinearity, not economics.
I drop a price that correlates > 0.9 with another and report the pair as not identified (15 of 60)
instead of publishing misleading numbers.

**7. Why time-based validation instead of a random split?**
A random split lets the model train on the future (week 380) to predict the past (week 379), and
lag features leak the answer. A rolling-origin backtest mimics production: train on everything up
to week T, forecast T+1..T+8, move T forward. I used 9 origins covering weeks 328-399, and chose
the LightGBM objective on a window *before* the first origin so the test folds were never used
for tuning.

**8. How did you make sure there's no leakage in the features?**
Every sales-based feature for a target week `t` at horizon `h` uses data up to `t-h` only (lags,
rolling stats). A unit test builds a series where units = week number and checks every lag equals
the expected week exactly for h = 1, 3, 8. Price/promo for the target week is used on purpose --
it is planned in advance -- and documented as an assumption.

**9. Isn't using the target week's price leakage?**
Not in this use case: retailers and bottlers plan price and promotions weeks ahead, and the
forecast is used to plan supply *given* that promo calendar. It does mean the reported accuracy is
"forecast given the plan"; if the plan changes, the forecast must be re-run. Without planned promo
information no model could predict a 15x promo spike.

**10. Why WAPE rather than MAPE?**
WAPE = total absolute error / total volume, so errors are weighted by volume -- what matters for
supply. MAPE averages per-row % errors and explodes on tiny weeks (forecast 5, actual 1 = 400%),
which dominate intermittent store-level data. I still report MAPE, bias and RMSE.

**11. Why did LightGBM beat the baseline in promo weeks but only modestly otherwise?**
In normal weeks demand is stable and the constant-elasticity regression describes it well (WAPE
0.449 vs LightGBM 0.400). In promo weeks the response depends on interactions -- depth x promo type x
pack x recent sales level x season -- that one elasticity can't capture; trees model them directly
(0.485 vs 0.598). Naive baselines fail in both because volume swings 4-16x between promo and
non-promo weeks.

**12. How did you estimate promotion lift?**
For each promo week in the backtest, the fold's LightGBM model (trained only on earlier data)
predicts the week as it happened and again with promotion removed and price reset to regular. The
difference to actual is the lift. Monotone constraints on price and discount keep that
counterfactual economically sensible. Sanity check: the baselines match actual non-promo volumes
closely.

**13. What did you find about promotions?**
They move huge volume (24-packs ~16x baseline) but on average don't pay back their discount (ROI
0.89). Depth is the lever: 10-20% discounts are margin-accretive (ROI 1.30-1.34) while 30%+
discounts lose money (0.72-0.82) and take about two-thirds of discount spend. Untagged price cuts
are the worst (ROI about 0.5). Recommendation: shift depth from 30%+ to 10-20%, and always pair cuts
with merchandising.

**14. If promotions lose money, why does the retailer run them?**
My ROI is category gross margin only. It ignores store traffic and basket size (soft drinks are a
classic loss leader), manufacturer funding not visible in acquisition cost, and competitive
positioning. A full trade-spend evaluation would add basket data and the bottler's trade terms.

**15. How would you handle a new product with no history?**
The global model helps: brand, pack, price and promo features transfer from similar SKUs. For
elasticity, borrow from analogous products -- pool with brand/pack-level hierarchical priors
(Bayesian shrinkage) and update as data arrives. Launch forecasts would rely on attribute-based
analogues plus planned distribution and promo.

**16. How would you productionize this forecast?**
Databricks: the PySpark ingest as a scheduled Workflow writing Delta tables; feature build and
LightGBM training as jobs tracked in MLflow (params, WAPE by fold, model artifact) and registered in
Unity Catalog; weekly batch scoring for the next 8 weeks with the promo plan as input; monitoring of
WAPE and bias by brand and promo vs non-promo, with alerts on drift; Power BI on top of the Delta
tables. My MLOps background (CI/CD, SageMaker, model registry) maps directly.

**17. How would you make the elasticities more robust?**
Hierarchical Bayesian model (store elasticities shrunk to brand-level means), instrumental
variables for endogeneity, separate short- vs long-run (regular price) elasticities, and
competitor-reaction models for scenario planning.

**18. Your optimiser says lower the Coke 2L regular price by 17%. Would you recommend that?**
Not on its own. It assumes constant elasticity, no competitor response and fixed cost, and the
elasticity comes from temporary deals, which overstate the reaction to a permanent change. I'd use
it to rank options and size the trade-off, then validate with a test-and-control store experiment
before any shelf-price change.

**19. What is cannibalisation and how did you measure it?**
Volume a promotion takes from other brands rather than adding to the category. I used the
cross-elasticities: a promotion of depth d on brand k changes competitor j's volume by
`(1-d)^{b_jk} - 1`. Coke 24-pack promotions take the most -- about 17% of competitors' volume.

**20. What would you do differently with more time?**
Instrument prices; hierarchical forecasting reconciled store -> chain; a promotion-calendar
optimiser under a trade budget; MLflow tracking on Databricks; and validate the counterfactual
lift against a holdout of stores that did not run a given promotion.

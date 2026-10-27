# Interview prep -- questions about this project

Short answers in my own words. Numbers come from `results/summary.json`.

**1. In one minute, what did you build?**
I took 17.7M rows of weekly supermarket scanner data for soft drinks, cleaned and aggregated it
with pandas, and built a LightGBM model that forecasts weekly sales per product and store 1-8 weeks
ahead. I tuned it with time-based cross-validation and tested it in a 9-fold rolling backtest
against three baselines: 48% lower error than "same week last year". I also estimated price
elasticities with a regression and used the forecast model to measure which promotions pay back.

**2. How did you handle 17.7M rows?**
pandas with explicit column types (small integer types, only the columns I need) -- it loads and
aggregates in under a minute on a laptop. After aggregating to brand x pack x store x week it is
2.4M rows, and the modelling table for 12 products is 381k rows.

**3. What cleaning did you do?**
From the dataset's codebook: unit price = bundle price / bundle size, drop rows flagged as suspect
and weeks with no price. I mapped 1,746 product descriptions to brands with ordered regex rules
(tested), flagged 1.2% of rows with impossible margins, and found 9 weeks with no data at all.

**4. How did you avoid data leakage?**
- Time-based validation only, never random splits.
- Sales features for a target week only use data up to the forecast origin; a unit test builds a
  series where units = week number and checks every lag for horizons 1, 3 and 8.
- The regular price only looks backwards (also tested).
- Hyperparameters were tuned on folds before the test period.

**5. Explain your validation setup.**
Rolling origin: train on everything up to week T, forecast weeks T+1 to T+8, move T forward 8 weeks,
9 times (test weeks 328-399). Tuning used 3 earlier folds (origins 303, 311, 319).

**6. How did you tune the model?**
8 settings from a small grid (Tweedie power, number of leaves, minimum rows per leaf), each
trained for up to 300 trees on the 3 tuning folds. I average the folds' validation error curves and
take the number of trees where the average is lowest. Best: Tweedie power 1.7, 127 leaves, 300
trees, CV WAPE 0.464 vs. 0.509 for the worst.

**6b. Did anything go wrong in tuning?**
Yes -- at first I averaged each fold's own early-stopping point. Re-running on numerically
identical data changed that from 93 to 68 trees, and the test WAPE from 0.475 to 0.491. Averaging
the whole validation curves first and then picking the minimum is much more stable.

**7. Why Tweedie loss?**
Sales are never negative and very skewed -- many normal weeks and some huge promotion weeks. Tweedie
is designed for that; squared error would be dominated by the biggest weeks.

**8. Why one global model instead of one per product?**
More data per model, and patterns (how promotions lift sales, holidays) are shared across
products. Product and store information enters as features.

**9. Isn't using the forecast week's price leakage?**
No -- prices and promotions are planned weeks ahead, so they are known when the forecast is made.
I state it as an assumption: the forecast is "sales given the promotion plan".

**10. Why WAPE and not MAPE?**
WAPE = total absolute error / total volume, so big weeks matter more, which is what matters for
supply. MAPE explodes on tiny weeks (forecast 5, actual 1 = 400% error). I report both.

**11. Why did the naive baselines do so badly?**
Volume jumps roughly 5-20x in promotion weeks. "Same week last year" is wrong whenever last year's
week was promoted and this year's isn't (or the reverse). The regression does well in normal weeks
but worse in promotion weeks, where LightGBM captures interactions between discount depth,
promotion type and recent sales.

**12. What does SHAP tell you?**
Discount depth is by far the biggest driver of the forecast, then the recent 13-week sales level and
the price -- you cannot forecast this category without the promotion plan.

**13. Why the monotone constraints?**
So the model cannot learn "higher price -> more sales" from noise. I use the model to answer "what
if there had been no promotion?", so the price effect has to point the right way.

**14. What is price elasticity and what did you find?**
The % change in volume for a 1% price change. A log-log regression gives it directly as the
coefficient on log price. All 12 products are very price-sensitive (-2.9 to -6.2), 24-packs more
than 2-litre bottles. Standard errors are clustered by store because weeks of one store are
correlated. Caveat: the retailer chooses when to discount, so the estimates are probably high.

**15. You found some elasticities couldn't be estimated -- why?**
Pepsi and Diet Pepsi 2L are always priced together (correlation 0.99), so the regression cannot
tell their effects apart; the first run gave nonsense numbers. I drop one of two prices that
correlate above 0.9 and report that pair as "not identified" instead of a misleading number.

**16. How did you measure promotion lift?**
The fold's model (trained only on earlier weeks) predicts each promoted week with and without the
promotion. Lift = actual - no-promotion prediction. ROI = extra margin from the lift / money given
away in discounts.

**17. Which promotions pay back?**
10-20% discounts return $1.31-1.39 per $1 of discount; 30%+ discounts only $0.72-0.83, yet they take
two-thirds of the discount budget.

**18. What would you do next?**
Forecast at chain level as well and reconcile it with store forecasts, test more model types, and
check the promotion baselines against stores that did not run a given promotion.

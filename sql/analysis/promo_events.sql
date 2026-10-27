-- Promotion events and their economics, from the out-of-sample forecasts in promo_counterfactual
-- (actual units + the model's "no promotion" baseline for every promoted store-week).
--
--   lift_units          = actual - baseline
--   discount_investment = (regular price - promo price) * actual units
--   incremental_gm      = lift_units * (regular price - unit cost)
--   net_margin_impact   = incremental_gm - discount_investment
--                       = actual gross margin - baseline gross margin at the regular price
--
-- Consecutive promoted weeks of the same store x series form one event ("gaps and islands":
-- week - ROW_NUMBER() is constant inside a run of consecutive weeks).
CREATE OR REPLACE TABLE promo_events AS
WITH promo_weeks AS (
    SELECT
        series, brand, pack, store, week, fold, promo_type,
        discount, unit_price, reg_price, unit_cost, units,
        pred_baseline                                            AS baseline_units,
        units - pred_baseline                                    AS lift_units,
        GREATEST(reg_price - unit_price, 0) * units              AS discount_investment,
        (units - pred_baseline) * (reg_price - unit_cost)        AS incremental_gm,
        units * unit_price - pred_baseline * reg_price           AS incremental_revenue,
        week - ROW_NUMBER() OVER (PARTITION BY series, store ORDER BY week) AS island
    FROM promo_counterfactual
    WHERE promo = 1
)
SELECT
    * EXCLUDE (island),
    incremental_gm - discount_investment                         AS net_margin_impact,
    DENSE_RANK() OVER (ORDER BY series, store, island)           AS event_id,
    CASE WHEN discount < 0.10 THEN '0-10%'
         WHEN discount < 0.20 THEN '10-20%'
         WHEN discount < 0.30 THEN '20-30%'
         ELSE '30%+' END                                         AS depth_bucket,
    CASE promo_type
         WHEN 'B' THEN 'Bonus buy (B)'
         WHEN 'S' THEN 'Simple price reduction (S)'
         WHEN 'C' THEN 'Coupon (C)'
         WHEN 'G' THEN 'Undocumented code (G)'
         WHEN 'U' THEN 'Unflagged price cut (U)' END             AS promo_label
FROM promo_weeks
ORDER BY series, store, week;

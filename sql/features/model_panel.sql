-- Modelling panel: one row per in-scope brand x pack x store x week, with price and promotion
-- features. Built in steps (temp tables) so each step can be inspected on its own.
--
-- Parameters: ${brands}, ${packs} (IN-lists), ${reg_preceding} (= regular-price window - 1), ${threshold} (min discount
-- that counts as a promotion), ${min_weeks} (min selling weeks per store x series).

-- Step 1: complete weekly grid for each store x series, from its first to its last sale.
-- Weeks with no data anywhere in the chain are "data gaps" (units stay NULL); other missing
-- weeks mean the item did not sell -> 0 units.
CREATE OR REPLACE TEMP TABLE p1_grid AS
WITH scope AS (
    SELECT * FROM fact_sales_weekly
    WHERE brand IN (${brands}) AND pack IN (${packs})
),
span AS (
    SELECT store, brand, pack, MIN(week) AS first_week, MAX(week) AS last_week
    FROM scope GROUP BY store, brand, pack
),
chain_weeks AS (SELECT DISTINCT week FROM fact_sales_weekly)
SELECT
    s.store, s.brand, s.pack, c.week,
    CASE WHEN cw.week IS NULL THEN 1 ELSE 0 END                           AS data_gap,
    CASE WHEN f.week IS NULL THEN 0 ELSE 1 END                            AS stocked,
    CASE WHEN cw.week IS NOT NULL THEN COALESCE(f.units, 0) END           AS units,
    CASE WHEN cw.week IS NOT NULL THEN COALESCE(f.dollars, 0) END         AS dollars,
    CASE WHEN cw.week IS NOT NULL THEN COALESCE(f.cogs, 0) END            AS cogs,
    CASE WHEN cw.week IS NOT NULL THEN COALESCE(f.eq_cases, 0) END        AS eq_cases,
    COALESCE(f.units_sale_b, 0) + 1e-6 * COALESCE(f.any_sale_b, 0)        AS score_b,
    COALESCE(f.units_sale_s, 0) + 1e-6 * COALESCE(f.any_sale_s, 0)        AS score_s,
    COALESCE(f.units_sale_c, 0) + 1e-6 * COALESCE(f.any_sale_c, 0)        AS score_c,
    COALESCE(f.units_sale_g, 0) + 1e-6 * COALESCE(f.any_sale_g, 0)        AS score_g,
    COALESCE(GREATEST(f.any_sale_b, f.any_sale_s, f.any_sale_c, f.any_sale_g), 0) AS has_sale_code,
    f.unit_price                                                          AS observed_price
FROM span s
JOIN dim_calendar c ON c.week BETWEEN s.first_week AND s.last_week
LEFT JOIN scope f ON f.store = s.store AND f.brand = s.brand AND f.pack = s.pack AND f.week = c.week
LEFT JOIN chain_weeks cw ON cw.week = c.week;

-- Step 2: price, regular price and discount.
--   unit_price: last known price carried forward (then backward at the start of a series).
--   reg_price : highest price in the current week and the ${reg_preceding} weeks before it -- only
--               past/current weeks, so a later price change cannot leak backwards.
CREATE OR REPLACE TEMP TABLE p2_price AS
WITH filled AS (
    SELECT *,
        COALESCE(
            observed_price,
            LAST_VALUE(observed_price IGNORE NULLS) OVER (
                PARTITION BY store, brand, pack ORDER BY week
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
            FIRST_VALUE(observed_price IGNORE NULLS) OVER (
                PARTITION BY store, brand, pack ORDER BY week
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
        ) AS unit_price
    FROM p1_grid
)
SELECT *,
    MAX(unit_price) OVER (
        PARTITION BY store, brand, pack ORDER BY week
        ROWS BETWEEN ${reg_preceding} PRECEDING AND CURRENT ROW) AS reg_price
FROM filled;

-- Step 3: promotion type, flags and post-promotion weeks.
--   The data's SALE code wins (dominant code by units: B bonus buy, S simple price reduction,
--   C coupon, G undocumented); otherwise a price cut >= ${threshold} is an unflagged promo (U).
CREATE OR REPLACE TEMP TABLE p3_promo AS
WITH typed AS (
    SELECT *,
        LEAST(GREATEST(1 - unit_price / reg_price, 0), 0.9) AS discount,
        CASE
            WHEN has_sale_code = 1 THEN
                CASE WHEN score_b >= score_s AND score_b >= score_c AND score_b >= score_g THEN 'B'
                     WHEN score_s >= score_c AND score_s >= score_g THEN 'S'
                     WHEN score_c >= score_g THEN 'C'
                     ELSE 'G' END
            WHEN LEAST(GREATEST(1 - unit_price / reg_price, 0), 0.9) >= ${threshold} THEN 'U'
            ELSE 'none'
        END AS promo_type
    FROM p2_price
),
flagged AS (
    SELECT *, CASE WHEN promo_type <> 'none' THEN 1 ELSE 0 END AS promo
    FROM typed
)
SELECT *,
    CASE WHEN promo_type = 'B' THEN 1 ELSE 0 END AS promo_b,
    CASE WHEN promo_type = 'S' THEN 1 ELSE 0 END AS promo_s,
    CASE WHEN promo_type = 'C' THEN 1 ELSE 0 END AS promo_c,
    CASE WHEN promo_type = 'G' THEN 1 ELSE 0 END AS promo_g,
    CASE WHEN promo_type = 'U' THEN 1 ELSE 0 END AS promo_u,
    CASE WHEN COALESCE(LAG(promo, 1) OVER w, 0) = 1 AND promo = 0 THEN 1 ELSE 0 END AS post_promo_1,
    CASE WHEN COALESCE(LAG(promo, 2) OVER w, 0) = 1 AND COALESCE(LAG(promo, 1) OVER w, 0) = 0
              AND promo = 0 THEN 1 ELSE 0 END AS post_promo_2
FROM flagged
WINDOW w AS (PARTITION BY store, brand, pack ORDER BY week);

-- Step 4: unit cost from the retailer margin. Implausible margins (outside -50%..90%) are
-- ignored and the last valid cost is carried forward (then backward).
CREATE OR REPLACE TEMP TABLE p4_cost AS
WITH raw_cost AS (
    SELECT *,
        CASE WHEN units > 0 AND dollars > 0
                  AND 1 - cogs / dollars BETWEEN -0.5 AND 0.9
             THEN cogs / units END AS cost_obs
    FROM p3_promo
)
SELECT *,
    COALESCE(
        cost_obs,
        LAST_VALUE(cost_obs IGNORE NULLS) OVER (
            PARTITION BY store, brand, pack ORDER BY week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
        FIRST_VALUE(cost_obs IGNORE NULLS) OVER (
            PARTITION BY store, brand, pack ORDER BY week
            ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
    ) AS unit_cost
FROM raw_cost;

-- Step 5: competitor prices. For every store x pack x week, the (log) price of each in-scope
-- brand in the same pack. A brand with no row that week gets its last known price in that store,
-- then its median price in that pack.
CREATE OR REPLACE TEMP TABLE p5_comp_long AS
WITH keys AS (SELECT DISTINCT store, pack, week FROM p4_cost),
brands AS (SELECT DISTINCT brand FROM p4_cost),
long AS (
    SELECT k.store, k.pack, k.week, b.brand, c.unit_price, c.discount
    FROM keys k
    CROSS JOIN brands b
    LEFT JOIN p4_cost c
           ON c.store = k.store AND c.pack = k.pack AND c.week = k.week AND c.brand = b.brand
),
filled AS (
    SELECT *,
        COALESCE(
            unit_price,
            LAST_VALUE(unit_price IGNORE NULLS) OVER (
                PARTITION BY store, pack, brand ORDER BY week
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
            FIRST_VALUE(unit_price IGNORE NULLS) OVER (
                PARTITION BY store, pack, brand ORDER BY week
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
        ) AS price_filled
    FROM long
)
SELECT store, pack, week, brand, discount,
    LN(COALESCE(price_filled, MEDIAN(price_filled) OVER (PARTITION BY pack, brand))) AS log_price
FROM filled;

-- one column per brand: lp_coke, lp_diet_coke, ...
CREATE OR REPLACE TEMP TABLE p5_comp_wide AS
PIVOT (
    SELECT store, pack, week, 'lp_' || LOWER(REPLACE(brand, ' ', '_')) AS col, log_price
    FROM p5_comp_long
) ON col USING FIRST(log_price) GROUP BY store, pack, week;

-- average discount of the *other* in-scope brands in the same pack and week
CREATE OR REPLACE TEMP TABLE p5_comp_discount AS
SELECT store, pack, week,
       SUM(discount) AS total_discount,
       COUNT(discount) AS n_brands
FROM p5_comp_long
GROUP BY store, pack, week;

-- Step 6: final panel. Keep store x series with >= ${min_weeks} selling weeks and add the
-- calendar and store attributes.
CREATE OR REPLACE TABLE model_panel AS
WITH selling AS (
    SELECT store, brand, pack
    FROM p4_cost
    WHERE units > 0
    GROUP BY store, brand, pack
    HAVING COUNT(*) >= ${min_weeks}
)
SELECT
    p.brand || ' | ' || p.pack                                   AS series,
    p.store, p.brand, p.pack, p.week,
    CAST(p.units AS DOUBLE) AS units,                            -- NULL in chain-wide data gaps
    p.dollars, p.cogs, p.eq_cases, p.data_gap, p.stocked,
    p.unit_price, p.reg_price, p.discount, p.unit_cost,
    LN(p.unit_price)                                             AS ln_price,
    LN(p.reg_price)                                              AS ln_reg_price,
    p.promo_type, p.promo, p.promo_b, p.promo_s, p.promo_c, p.promo_g, p.promo_u,
    p.post_promo_1, p.post_promo_2,
    w.* EXCLUDE (store, pack, week),
    COALESCE((d.total_discount - COALESCE(p.discount, 0)) / GREATEST(d.n_brands - 1, 1), 0)
                                                                 AS comp_discount,
    c.* EXCLUDE (week),
    s.city, s.zone, COALESCE(s.price_tier, 'Unknown')            AS price_tier,
    s.lat, s.lon, s.income_log_median, s.pct_college, s.pct_age60plus, s.pct_ethnic,
    s.avg_household_size, s.pop_density
FROM p4_cost p
JOIN selling USING (store, brand, pack)
JOIN p5_comp_wide w     ON w.store = p.store AND w.pack = p.pack AND w.week = p.week
JOIN p5_comp_discount d ON d.store = p.store AND d.pack = p.pack AND d.week = p.week
JOIN dim_calendar c     ON c.week = p.week
LEFT JOIN dim_store s   ON s.store = p.store
ORDER BY series, p.store, p.week;

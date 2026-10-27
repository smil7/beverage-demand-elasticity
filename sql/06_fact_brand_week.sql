-- Coarser chain-level table: brand x week, volume in 8-oz equivalent cases (192 oz) so packs
-- of different sizes can be added up.
CREATE OR REPLACE TEMP VIEW fact_brand_week AS
SELECT
    f.brand,
    f.week,
    c.week_start,
    SUM(f.eq_cases)                         AS eq_cases,
    SUM(f.units)                            AS units,
    SUM(f.dollars)                          AS dollars,
    SUM(f.dollars) - SUM(f.cogs)            AS gross_margin,
    SUM(f.dollars) / NULLIF(SUM(f.eq_cases), 0) AS price_per_eq_case,
    COUNT(DISTINCT f.store)                 AS n_stores
FROM fact_sales_weekly f
JOIN dim_calendar c ON f.week = c.week
GROUP BY f.brand, f.week, c.week_start;

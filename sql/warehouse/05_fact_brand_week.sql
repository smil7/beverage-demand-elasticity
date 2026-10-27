-- Chain-level brand x week table. Packs of different sizes are added up in
-- 8-oz equivalent cases so volumes are comparable.
CREATE OR REPLACE TABLE fact_brand_week AS
SELECT
    f.brand,
    f.week,
    c.week_start,
    SUM(f.eq_cases)                               AS eq_cases,
    SUM(f.units)                                  AS units,
    SUM(f.dollars)                                AS dollars,
    SUM(f.dollars) - SUM(f.cogs)                  AS gross_margin,
    SUM(f.dollars) / NULLIF(SUM(f.eq_cases), 0)   AS price_per_eq_case,
    COUNT(DISTINCT f.store)                       AS n_stores
FROM fact_sales_weekly f
JOIN dim_calendar c ON f.week = c.week
GROUP BY f.brand, f.week, c.week_start
ORDER BY f.brand, f.week;

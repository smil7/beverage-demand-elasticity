-- Modelling grain: brand x pack x store x week.
-- Price is the volume-weighted average unit price (dollars / units); weeks with zero units fall
-- back to the simple average shelf price of the UPCs on sale that week.
CREATE OR REPLACE TEMP VIEW fact_sales_weekly AS
SELECT
    m.store,
    p.brand,
    p.pack,
    m.week,
    SUM(m.units)                                             AS units,
    SUM(m.dollars)                                           AS dollars,
    SUM(m.cogs)                                              AS cogs,
    SUM(m.units * p.oz_per_unit) / 192.0                     AS eq_cases,
    CASE WHEN SUM(m.units) > 0 THEN SUM(m.dollars) / SUM(m.units)
         ELSE AVG(m.unit_price) END                          AS unit_price,
    SUM(CASE WHEN m.sale_code = 'B' THEN m.units ELSE 0 END) AS units_sale_b,
    SUM(CASE WHEN m.sale_code = 'S' THEN m.units ELSE 0 END) AS units_sale_s,
    SUM(CASE WHEN m.sale_code = 'C' THEN m.units ELSE 0 END) AS units_sale_c,
    SUM(CASE WHEN m.sale_code = 'G' THEN m.units ELSE 0 END) AS units_sale_g,
    MAX(CASE WHEN m.sale_code = 'B' THEN 1 ELSE 0 END)       AS any_sale_b,
    MAX(CASE WHEN m.sale_code = 'S' THEN 1 ELSE 0 END)       AS any_sale_s,
    MAX(CASE WHEN m.sale_code = 'C' THEN 1 ELSE 0 END)       AS any_sale_c,
    MAX(CASE WHEN m.sale_code = 'G' THEN 1 ELSE 0 END)       AS any_sale_g,
    COUNT(DISTINCT m.upc)                                    AS n_upcs
FROM movement_clean m
JOIN dim_product p ON m.upc = p.upc
GROUP BY m.store, p.brand, p.pack, m.week;

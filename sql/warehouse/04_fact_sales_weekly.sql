-- Fact table at the modelling grain: brand x pack x store x week.
-- The primary key makes the database reject duplicate keys.
CREATE OR REPLACE TABLE fact_sales_weekly (
    store         INTEGER NOT NULL,
    brand         VARCHAR NOT NULL,
    pack          VARCHAR NOT NULL,
    week          INTEGER NOT NULL,
    units         BIGINT  NOT NULL,
    dollars       DOUBLE  NOT NULL,
    cogs          DOUBLE,
    eq_cases      DOUBLE,              -- 8-oz equivalent cases (192 oz)
    unit_price    DOUBLE  NOT NULL,    -- volume-weighted average price per unit
    units_sale_b  BIGINT, units_sale_s BIGINT, units_sale_c BIGINT, units_sale_g BIGINT,
    any_sale_b    INTEGER, any_sale_s INTEGER, any_sale_c INTEGER, any_sale_g INTEGER,
    n_upcs        INTEGER,
    PRIMARY KEY (store, brand, pack, week)
);

INSERT INTO fact_sales_weekly
SELECT
    m.store,
    p.brand,
    p.pack,
    m.week,
    SUM(m.units)                                             AS units,
    SUM(m.dollars)                                           AS dollars,
    SUM(m.cogs)                                              AS cogs,
    SUM(m.units * p.oz_per_unit) / 192.0                     AS eq_cases,
    -- weeks with zero units fall back to the simple average shelf price
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

-- Data-quality checks on the warehouse. One row per check:
--   failing_rows = rows that violate the rule; blocking = the pipeline stops if it fails.
WITH f AS (SELECT * FROM fact_sales_weekly),
n AS (SELECT COUNT(*) AS total FROM f),
week_range AS (SELECT MIN(week) AS lo, MAX(week) AS hi FROM f),
missing_weeks AS (
    SELECT c.week
    FROM dim_calendar c, week_range r
    WHERE c.week BETWEEN r.lo AND r.hi
      AND c.week NOT IN (SELECT DISTINCT week FROM f)
),
store_weeks AS (SELECT store, COUNT(DISTINCT week) AS n_weeks FROM f GROUP BY store)

SELECT 'No duplicate brand x pack x store x week keys' AS check_name, TRUE AS blocking,
       (SELECT COUNT(*) - COUNT(DISTINCT (store, brand, pack, week)) FROM f) AS failing_rows,
       0.0 AS max_share, '' AS detail
UNION ALL
SELECT 'No negative volumes', TRUE, (SELECT COUNT(*) FROM f WHERE units < 0), 0.0, ''
UNION ALL
SELECT 'Unit price strictly positive', TRUE,
       (SELECT COUNT(*) FROM f WHERE unit_price IS NULL OR unit_price <= 0), 0.0, ''
UNION ALL
SELECT 'Unit price sane (<= $25 per pack)', FALSE,
       (SELECT COUNT(*) FROM f WHERE unit_price > 25), 0.001, ''
UNION ALL
SELECT 'Every UPC sold is in dim_product', TRUE,
       (SELECT COUNT(*) FROM movement_clean m
        WHERE NOT EXISTS (SELECT 1 FROM dim_product p WHERE p.upc = m.upc)), 0.0, ''
UNION ALL
SELECT 'Every week maps to the calendar', TRUE,
       (SELECT COUNT(*) FROM f WHERE week NOT IN (SELECT week FROM dim_calendar)), 0.0, ''
UNION ALL
SELECT 'Every store is in dim_store', FALSE,
       (SELECT COUNT(DISTINCT store) FROM f WHERE store NOT IN (SELECT store FROM dim_store)), 0.0,
       'stores without a demographics row; they get price tier ''Unknown'''
UNION ALL
SELECT 'Gross margin within [-50%, 90%]', FALSE,
       (SELECT COUNT(*) FROM f
        WHERE dollars > 0 AND (1 - cogs / dollars < -0.5 OR 1 - cogs / dollars > 0.9)), 0.01,
       'kept but flagged; the model panel ignores these costs'
UNION ALL
SELECT 'Chain-wide week coverage (<= 12 empty weeks)', FALSE,
       GREATEST((SELECT COUNT(*) FROM missing_weeks) - 12, 0), 0.0,
       'empty weeks: ' || (SELECT COALESCE(STRING_AGG(week::VARCHAR, ', ' ORDER BY week), 'none')
                           FROM missing_weeks)
UNION ALL
SELECT 'Stores with < 52 weeks of data', FALSE,
       0, 0.0,
       (SELECT COUNT(*) FILTER (WHERE n_weeks < 52) || ' of ' || COUNT(*) || ' stores; min '
               || MIN(n_weeks) || ', median ' || MEDIAN(n_weeks)::INT || ' weeks' FROM store_weeks);

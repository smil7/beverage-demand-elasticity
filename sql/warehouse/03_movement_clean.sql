-- Cleaned UPC x store x week movement (a view over staging).
-- Rules from the Dominick's codebook:
--   * PRICE is the price of a bundle of QTY items and MOVE counts items, so
--     unit price = PRICE / QTY and dollar sales = PRICE * MOVE / QTY.
--   * PROFIT is the retailer's gross margin in %, so cost of goods = dollars * (1 - PROFIT/100).
--   * OK = 0 marks suspect data; PRICE <= 0 means the item was not on sale that week.
CREATE OR REPLACE VIEW movement_clean AS
SELECT
    STORE                       AS store,
    UPC                         AS upc,
    WEEK                        AS week,
    MOVE                        AS units,
    QTY                         AS qty,
    PRICE / QTY                 AS unit_price,
    PRICE * MOVE / QTY          AS dollars,
    PROFIT                      AS margin_pct,
    -- margins outside [-100, 100] are recording errors: treat the cost as unknown
    CASE WHEN PROFIT BETWEEN -100 AND 100
         THEN PRICE * MOVE / QTY * (1 - PROFIT / 100) END AS cogs,
    NULLIF(TRIM(SALE), '')      AS sale_code
FROM stg_movement
WHERE OK = 1
  AND PRICE > 0
  AND MOVE >= 0
  AND QTY > 0;

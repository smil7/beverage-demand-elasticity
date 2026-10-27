-- Cleaned UPC x store x week movement.
-- Rules from the Dominick's codebook (Part 4):
--   * PRICE is the price of a bundle of QTY items; MOVE counts items, so unit price = PRICE / QTY
--     and dollar sales = PRICE * MOVE / QTY.
--   * PROFIT is the retailer's gross margin in percent, so cost of goods = dollars * (1 - PROFIT/100).
--   * OK = 0 marks suspect data; PRICE <= 0 means the item was not on sale/stocked that week.
CREATE OR REPLACE TEMP VIEW movement_clean AS
SELECT
    CAST(STORE AS INT)                              AS store,
    CAST(UPC AS BIGINT)                             AS upc,
    CAST(WEEK AS INT)                               AS week,
    CAST(MOVE AS BIGINT)                            AS units,
    CAST(QTY AS INT)                                AS qty,
    PRICE / QTY                                     AS unit_price,
    PRICE * MOVE / QTY                              AS dollars,
    PROFIT                                          AS margin_pct,
    -- margins outside [-100, 100] are recording errors; treat them as unknown (NULL cost)
    CASE WHEN PROFIT BETWEEN -100 AND 100
         THEN PRICE * MOVE / QTY * (1 - PROFIT / 100) END AS cogs,
    NULLIF(TRIM(SALE), '')                          AS sale_code
FROM raw_movement
WHERE OK = 1
  AND PRICE > 0
  AND MOVE >= 0
  AND QTY > 0;

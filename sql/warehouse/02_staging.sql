-- Staging: load the raw movement CSV (17.7M rows) as-is, with explicit column types.
CREATE OR REPLACE TABLE stg_movement AS
SELECT *
FROM read_csv(
    '${movement_csv}',
    header = true,
    columns = {
        'STORE': 'INTEGER', 'UPC': 'BIGINT', 'WEEK': 'INTEGER', 'MOVE': 'BIGINT',
        'QTY': 'INTEGER', 'PRICE': 'DOUBLE', 'SALE': 'VARCHAR', 'PROFIT': 'DOUBLE',
        'OK': 'INTEGER', 'PRICE_HEX': 'VARCHAR', 'PROFIT_HEX': 'VARCHAR'
    }
);

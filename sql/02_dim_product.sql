-- One row per UPC with the derived brand / pack hierarchy (rules in config/brand_mapping.csv).
CREATE OR REPLACE TEMP VIEW dim_product AS
SELECT
    upc,
    description,
    size,
    brand,
    pack,
    items_per_pack,
    oz_per_item,
    oz_per_unit,
    brand || ' | ' || pack AS series
FROM dim_product_src;

-- Store attributes: pricing zone, price tier and 1990-census demographics of the trading area.
CREATE OR REPLACE TEMP VIEW dim_store AS
SELECT
    store,
    city,
    zone,
    price_tier,
    lat,
    lon,
    income_log_median,
    pct_college,
    pct_age60plus,
    pct_ethnic,
    avg_household_size,
    pop_density
FROM dim_store_src;

-- Star schema: dimension tables with primary keys.
-- Dimensions are small and built in Python from the codebook rules (brand regexes, calendar,
-- store demographics), then INSERTed here so the database enforces their keys.

CREATE OR REPLACE TABLE dim_product (
    upc             BIGINT PRIMARY KEY,
    description     VARCHAR,
    size            VARCHAR,
    case_pack       INTEGER,
    com_code        INTEGER,
    brand           VARCHAR NOT NULL,   -- from config/brand_mapping.csv
    pack            VARCHAR NOT NULL,   -- e.g. '2L', '24pk can'
    items_per_pack  INTEGER,
    oz_per_item     DOUBLE,
    oz_per_unit     DOUBLE
);

CREATE OR REPLACE TABLE dim_store (
    store               INTEGER PRIMARY KEY,
    city                VARCHAR,
    zone                INTEGER,
    price_tier          VARCHAR NOT NULL,   -- Low / Medium / High / Unknown
    lat                 DOUBLE,
    lon                 DOUBLE,
    income_log_median   DOUBLE,
    pct_college         DOUBLE,
    pct_age60plus       DOUBLE,
    pct_ethnic          DOUBLE,
    avg_household_size  DOUBLE,
    pop_density         DOUBLE
);

CREATE OR REPLACE TABLE dim_calendar (
    week                INTEGER PRIMARY KEY,   -- Dominick's week number, week 1 = 1989-09-14
    week_start          DATE NOT NULL,
    week_end            DATE NOT NULL,
    year                INTEGER,
    month               INTEGER,
    quarter             INTEGER,
    week_of_year        INTEGER,
    hol_new_year        INTEGER,
    hol_super_bowl      INTEGER,
    hol_presidents_day  INTEGER,
    hol_easter          INTEGER,
    hol_memorial_day    INTEGER,
    hol_july_4          INTEGER,
    hol_labor_day       INTEGER,
    hol_halloween       INTEGER,
    hol_thanksgiving    INTEGER,
    hol_christmas       INTEGER,
    is_holiday          INTEGER,
    holiday_name        VARCHAR,
    sin1 DOUBLE, cos1 DOUBLE, sin2 DOUBLE, cos2 DOUBLE, sin3 DOUBLE, cos3 DOUBLE
);

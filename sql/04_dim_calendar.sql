-- Dominick's week -> calendar dates, holidays and seasonality terms.
CREATE OR REPLACE TEMP VIEW dim_calendar AS
SELECT * FROM dim_calendar_src;

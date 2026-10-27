-- Forecast accuracy by model and ${group_by}, computed in the database from forecast_predictions
-- (long format: one row per series x store x week x model) or its chain-level rollup.
--   WAPE = sum |error| / sum actual      (volume-weighted error, the main metric)
--   bias = sum (forecast - actual) / sum actual   (+ = over-forecasting)
SELECT
    ${group_by},
    model,
    COUNT(*)                                                     AS n,
    SUM(ABS(units - forecast)) / SUM(units)                      AS wape,
    AVG(ABS(units - forecast) / units) FILTER (WHERE units > 0)  AS mape,
    SUM(forecast - units) / SUM(units)                           AS bias,
    SQRT(AVG((units - forecast) ^ 2))                            AS rmse
FROM ${source}
GROUP BY ${group_by}, model
ORDER BY 1, wape;

-- Promotion lift and ROI aggregated by ${group_by} (e.g. "pack, depth_bucket").
-- ROI = incremental gross margin / discount investment; ROI > 1 means the promotion added margin.
SELECT
    ${group_by},
    COUNT(*)                                              AS promo_weeks,
    COUNT(DISTINCT event_id)                              AS events,
    AVG(discount)                                         AS avg_discount,
    SUM(units)                                            AS actual_units,
    SUM(baseline_units)                                   AS baseline_units,
    SUM(lift_units)                                       AS lift_units,
    SUM(discount_investment)                              AS discount_investment,
    SUM(incremental_gm)                                   AS incremental_gm,
    SUM(net_margin_impact)                                AS net_margin_impact,
    SUM(incremental_revenue)                              AS incremental_revenue,
    SUM(lift_units) / SUM(baseline_units)                 AS lift_pct,
    SUM(incremental_gm) / NULLIF(SUM(discount_investment), 0) AS roi,
    COALESCE(SUM(incremental_gm) / NULLIF(SUM(discount_investment), 0) > 1, FALSE) AS pays_back
FROM promo_events
GROUP BY ${group_by}
HAVING COUNT(*) >= ${min_weeks} AND AVG(discount) >= ${min_discount}
ORDER BY ${group_by};

"""Promotion economics SQL (sql/analysis/promo_events.sql, promo_roi.sql) on a tiny table."""

import pandas as pd
import pytest

from beverage_demand.db import connect, run_sql_file, write_table
from beverage_demand.promo.analysis import roi_by


@pytest.fixture
def con():
    # one store x series: weeks 1-2 one promo event, week 3 no promo, week 4 a second event
    cf = pd.DataFrame({
        "series": "Coke | 2L", "brand": "Coke", "pack": "2L", "store": 1, "fold": 0,
        "week": [1, 2, 3, 4],
        "promo": [1, 1, 0, 1],
        "promo_type": ["S", "S", "none", "B"],
        "units": [150.0, 120.0, 90.0, 100.0],
        "pred_actual": [140.0, 118.0, 95.0, 101.0],
        "pred_baseline": [100.0, 100.0, 95.0, 100.0],
        "reg_price": [2.0, 2.0, 2.0, 2.0],
        "unit_price": [1.5, 1.6, 2.0, 2.0],
        "discount": [0.25, 0.20, 0.0, 0.0],
        "unit_cost": [1.2, 1.2, 1.2, 1.2],
    })
    c = connect(":memory:")
    write_table(c, "promo_counterfactual", cf)
    run_sql_file(c, "analysis/promo_events.sql")
    return c


def test_promo_week_economics(con):
    p = con.execute("SELECT * FROM promo_events").df().set_index("week")
    assert 3 not in p.index  # non-promo weeks are excluded
    w1 = p.loc[1]
    assert w1["lift_units"] == 50
    assert w1["discount_investment"] == pytest.approx(0.5 * 150)
    assert w1["incremental_gm"] == pytest.approx(50 * 0.8)
    # net impact == actual gross margin - baseline gross margin at the regular price
    assert w1["net_margin_impact"] == pytest.approx(150 * (1.5 - 1.2) - 100 * (2.0 - 1.2))
    assert w1["depth_bucket"] == "20-30%"
    assert w1["promo_label"] == "Simple price reduction (S)"


def test_events_gaps_and_islands(con):
    p = con.execute("SELECT week, event_id FROM promo_events ORDER BY week").df()
    assert p["event_id"].tolist() == [1, 1, 2]  # weeks 1-2 = one event, week 4 = another


def test_roi_aggregation(con):
    s = roi_by(con, "series").iloc[0]
    inc_gm = (50 + 20 + 0) * 0.8
    invest = 0.5 * 150 + 0.4 * 120 + 0.0
    assert s["roi"] == pytest.approx(inc_gm / invest)
    assert s["lift_pct"] == pytest.approx(70 / 300)
    assert s["events"] == 2
    assert not s["pays_back"]
    # the HAVING filter drops groups below the minimum discount
    assert roi_by(con, "series", min_discount=0.5).empty

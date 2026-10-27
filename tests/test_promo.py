import pandas as pd
import pytest

from beverage_demand.promo.analysis import label_events, promo_weeks, summarise

EDGES = [0.0, 0.1, 0.2, 0.3, 1.0]


@pytest.fixture
def cf():
    # one store x series: weeks 1-2 one promo event, week 3 no promo, week 4 a second event
    return pd.DataFrame({
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


def test_promo_week_economics(cf):
    p = promo_weeks(cf, EDGES).set_index("week")
    w1 = p.loc[1]
    assert w1["lift_units"] == 50
    assert w1["discount_investment"] == pytest.approx(0.5 * 150)
    assert w1["incremental_gm"] == pytest.approx(50 * 0.8)
    # net impact == actual gross margin - baseline gross margin at the regular price
    actual_gm = 150 * (1.5 - 1.2)
    baseline_gm = 100 * (2.0 - 1.2)
    assert w1["net_margin_impact"] == pytest.approx(actual_gm - baseline_gm)
    assert w1["depth_bucket"] == "20-30%"
    assert 3 not in p.index  # non-promo weeks are excluded


def test_events_and_roi(cf):
    p = label_events(promo_weeks(cf, EDGES))
    assert p["event_id"].nunique() == 2
    s = summarise(p, ["series"]).iloc[0]
    inc_gm = (50 + 20 + 0) * 0.8
    invest = 0.5 * 150 + 0.4 * 120 + 0.0
    assert s["roi"] == pytest.approx(inc_gm / invest)
    assert s["lift_pct"] == pytest.approx(70 / 300)
    assert not s["pays_back"]

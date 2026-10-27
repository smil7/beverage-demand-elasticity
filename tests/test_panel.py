"""Tests for the SQL feature panel (sql/features/model_panel.sql) on tiny in-memory databases."""

import numpy as np
import pytest
from sqlutil import build_panel_sql, db_with_fact, fact_rows


def test_regular_price_discount_and_promo_type():
    rows = fact_rows(1, "Coke", "2L", [1, 2, 3, 4, 5, 6], [2.0, 2.0, 1.5, 2.0, 1.9, 2.0],
                     codes=["", "", "B", "", "", ""])
    p = build_panel_sql(db_with_fact(rows), ["Coke"], ["2L"])
    assert list(p["reg_price"]) == [2.0] * 6
    assert np.isclose(p["discount"].iloc[2], 0.25)
    # week 3 has a SALE code B; week 5 is an unflagged 5% cut -> U
    assert list(p["promo_type"]) == ["none", "none", "B", "none", "U", "none"]
    assert list(p["post_promo_1"]) == [0, 0, 0, 1, 0, 1]
    assert list(p["post_promo_2"]) == [0, 0, 0, 0, 0, 0]  # week 5 is itself a promo
    assert np.allclose(p["unit_cost"], p["unit_price"] * 0.75)


def test_regular_price_uses_only_past_and_current_weeks():
    # a price rise in week 4 must not change the regular price of weeks 1-3 (no leakage)
    rows = fact_rows(1, "Coke", "2L", [1, 2, 3, 4, 5], [2.0, 2.0, 2.0, 3.0, 3.0])
    p = build_panel_sql(db_with_fact(rows), ["Coke"], ["2L"])
    assert list(p["reg_price"]) == [2.0, 2.0, 2.0, 3.0, 3.0]


def test_grid_fills_missing_weeks_and_marks_chain_gaps():
    # store 1 sells weeks 1, 2 and 5; store 2 sells weeks 1, 2, 4, 5 -> week 3 has no data anywhere
    rows = (fact_rows(1, "Coke", "2L", [1, 2, 5], [2.0, 2.0, 2.0])
            + fact_rows(2, "Coke", "2L", [1, 2, 4, 5], [2.0, 2.0, 2.0, 2.0]))
    p = build_panel_sql(db_with_fact(rows), ["Coke"], ["2L"]).set_index(["store", "week"])
    assert p.loc[(1, 3), "data_gap"] == 1 and np.isnan(p.loc[(1, 3), "units"])
    assert p.loc[(1, 4), "data_gap"] == 0 and p.loc[(1, 4), "units"] == 0  # not sold -> 0
    assert p.loc[(1, 4), "stocked"] == 0
    assert p.loc[(1, 4), "unit_price"] == 2.0  # last known price carried forward


def test_competitor_prices_and_discount():
    rows = (fact_rows(1, "Coke", "2L", [1, 2, 3], [2.0, 1.0, 2.0])
            + fact_rows(1, "Pepsi", "2L", [1, 3], [1.5, 1.5]))  # Pepsi missing in week 2
    p = build_panel_sql(db_with_fact(rows), ["Coke", "Pepsi"], ["2L"])
    coke = p[p["brand"] == "Coke"].set_index("week")
    assert np.allclose(np.exp(coke["lp_pepsi"]), 1.5)        # week 2 carried forward
    assert np.allclose(np.exp(coke["lp_coke"]), coke["unit_price"])
    pepsi = p[p["brand"] == "Pepsi"].set_index("week")
    assert pepsi.loc[2, "comp_discount"] == pytest.approx(0.5)  # Coke was 50% off in week 2


def test_min_weeks_filter():
    rows = fact_rows(1, "Coke", "2L", [1, 2, 3], [2.0] * 3) + fact_rows(2, "Coke", "2L", [1], [2.0])
    p = build_panel_sql(db_with_fact(rows), ["Coke"], ["2L"], min_weeks=2)
    assert set(p["store"]) == {1}

import numpy as np
import pandas as pd

from beverage_demand.features.panel import add_price_promo, complete_grid


def _series(prices, codes=None, units=None):
    n = len(prices)
    codes = codes or [""] * n
    units = units or [10] * n
    df = pd.DataFrame({"store": 1, "brand": "Coke", "pack": "2L", "week": np.arange(1, n + 1),
                       "unit_price": prices, "units": units, "dollars": np.multiply(prices, units),
                       "cogs": np.multiply(prices, units) * 0.75})
    for c in "bscg":
        df[f"units_sale_{c}"] = [u if code.lower() == c else 0 for u, code in zip(units, codes, strict=True)]
        df[f"any_sale_{c}"] = [int(code.lower() == c) for code in codes]
    return df


def test_regular_price_discount_and_promo_type():
    df = add_price_promo(_series([2.0, 2.0, 1.5, 2.0, 1.9, 2.0], ["", "", "B", "", "", ""]), 13, 0.05)
    assert list(df["reg_price"]) == [2.0] * 6
    assert np.isclose(df["discount"].iloc[2], 0.25)
    # week 3 flagged B; week 5 is a 5% unflagged cut -> U
    assert list(df["promo_type"]) == ["none", "none", "B", "none", "U", "none"]
    assert list(df["post_promo_1"]) == [0, 0, 0, 1, 0, 1]
    assert list(df["post_promo_2"]) == [0, 0, 0, 0, 0, 0]  # week 5 is itself a promo
    assert np.allclose(df["unit_cost"], df["unit_price"] * 0.75)


def test_regular_price_uses_only_past_and_current_weeks():
    # a price rise in week 4 must not change the regular price of weeks 1-3
    df = add_price_promo(_series([2.0, 2.0, 2.0, 3.0, 3.0]), 13, 0.05)
    assert list(df["reg_price"]) == [2.0, 2.0, 2.0, 3.0, 3.0]


def test_complete_grid_marks_gaps_and_zero_sales():
    f = _series([2.0, 2.0, 2.0]).iloc[[0, 2]]  # week 2 missing for this item
    out = complete_grid(f.assign(week=[1, 4]), chain_weeks={1, 2, 4})  # week 3: chain-wide gap
    out = out.set_index("week")
    assert out.loc[2, "units"] == 0 and out.loc[2, "stocked"] == 0
    assert np.isnan(out.loc[3, "units"]) and out.loc[3, "data_gap"] == 1

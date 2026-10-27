import numpy as np
import pytest

from beverage_demand.forecasting.backtest import counterfactual
from beverage_demand.forecasting.features import STORE_ATTRS, build_supervised
from beverage_demand.forecasting.metrics import bias, mape, wape

BRANDS = ["Coke", "Pepsi"]


@pytest.fixture(scope="module")
def sup(synthetic_panel):
    p = synthetic_panel[synthetic_panel["store"] <= 2].copy()
    # units = week number makes every lag directly checkable
    p["units"] = p["week"].astype(float)
    p["ln_reg_price"] = np.log(p["reg_price"])
    p["lp_coke"] = p["ln_price"]
    p["comp_discount"] = 0.0
    p["unit_cost"] = p["unit_price"] * 0.75
    for c in STORE_ATTRS:
        p[c] = 1.0
    return build_supervised(p, BRANDS, horizon=8, lags=[1, 2, 4, 8])


@pytest.mark.parametrize("h", [1, 3, 8])
def test_lags_only_use_data_before_the_origin(sup, h):
    d = sup[(sup["h"] == h) & (sup["week"] > 70)]
    for k in (1, 2, 4, 8):
        assert np.allclose(d[f"lag_{k}"], d["week"] - (h + k - 1))
    assert np.allclose(d["lag_52"], d["week"] - 52)
    # rolling mean of the 4 weeks ending at the origin t - h
    assert np.allclose(d["rmean_4"], d["week"] - h - 1.5)
    # no sales feature may reveal the target week or anything after the origin
    sales_feats = [c for c in d.columns if c.startswith(("lag_", "rmean", "rstd"))]
    assert (d[[c for c in sales_feats if c.startswith("lag_")]].max(axis=1) <= d["origin_week"]).all()


def test_origin_week(sup):
    assert (sup["origin_week"] == sup["week"] - sup["h"]).all()


def test_counterfactual_removes_promo(sup):
    cf = counterfactual(sup.head(200))
    assert (cf["discount"] == 0).all()
    assert (cf[["promo_b", "promo_s", "promo_c", "promo_g", "promo_u"]] == 0).all().all()
    assert np.allclose(cf["ln_price"], cf["ln_reg_price"])


def test_metrics():
    y = np.array([10.0, 0.0, 30.0])
    p = np.array([12.0, 1.0, 27.0])
    assert wape(y, p) == pytest.approx(6 / 40)
    assert mape(y, p) == pytest.approx((0.2 + 0.1) / 2)  # zero-actual week excluded
    assert bias(y, p) == pytest.approx(0.0)

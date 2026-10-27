import numpy as np
from fixtures.synthetic import BRANDS, TRUE_CROSS, TRUE_OWN, TRUE_PROMO_LOG_LIFT

from beverage_demand.elasticity.model import fit_series, within_store_corr


def test_recovers_known_own_and_cross_elasticity(synthetic_panel):
    m = fit_series(synthetic_panel, "Coke", BRANDS)
    p = m.result.params
    assert abs(p["ln_price"] - TRUE_OWN) < 0.15
    assert abs(p["lp_pepsi"] - TRUE_CROSS) < 0.1
    assert abs(p["promo_s"] - TRUE_PROMO_LOG_LIFT) < 0.1
    lo, hi = m.result.conf_int().loc["ln_price"]
    assert lo < TRUE_OWN < hi


def test_trend_variant_predicts_positive_units(synthetic_panel):
    m = fit_series(synthetic_panel, "Coke", BRANDS, time_effects="trend")
    pred = m.predict_units(synthetic_panel.head(50))
    assert np.all(pred > 0)
    # predictions should be in the right ballpark of actual volume
    ratio = pred.sum() / synthetic_panel.head(50)["units"].sum()
    assert 0.8 < ratio < 1.25


def test_lockstep_competitor_is_dropped(synthetic_panel):
    df = synthetic_panel.copy()
    df["lp_pepsi"] = df["ln_price"] + 0.001  # competitor priced exactly like the focal brand
    assert within_store_corr(df, "ln_price", "lp_pepsi") > 0.99
    m = fit_series(df, "Coke", BRANDS)
    assert "lp_pepsi" not in m.result.params
    assert m.spec.collinear_brands == ["Pepsi"]


def test_tier_interactions(synthetic_panel):
    m = fit_series(synthetic_panel, "Coke", BRANDS, tier_interactions=True)
    for t in ("High", "Low"):
        assert abs(m.result.params[f"ln_price_x_{t}"] - TRUE_OWN) < 0.25

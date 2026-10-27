"""Margin-maximising regular price within business bounds (constant-elasticity demand).

    volume(p) = Q0 * (p / p0) ** e
    margin(p) = (p - c) * volume(p)

With e < -1 the unconstrained optimum is p* = c * e / (1 + e) (the Lerner rule); ``scipy`` is used
so the same code works with bounds and other objectives. Competitor prices are held fixed and the
retailer's unit cost is assumed constant -- both are strong assumptions, see docs/methodology.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from beverage_demand.config import results, settings
from beverage_demand.scenario.simulate import baseline
from beverage_demand.viz import plots
from beverage_demand.viz.plots import plt


def margin_curve(p: np.ndarray, p0: float, q0: float, cost: float, e: float) -> np.ndarray:
    return (p - cost) * q0 * (p / p0) ** e


def optimal_price(p0: float, q0: float, cost: float, e: float, bound: float,
                  objective: str = "margin") -> float:
    def neg(p: float) -> float:
        vol = q0 * (p / p0) ** e
        return -((p - cost) * vol if objective == "margin" else p * vol)

    res = minimize_scalar(neg, bounds=(p0 * (1 - bound), p0 * (1 + bound)), method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x)


def optimise_all() -> pd.DataFrame:
    bound = settings()["scenario"]["price_bounds_pct"]
    own = pd.read_csv(results("elasticities.csv"))
    rows = []
    for _, r in own.iterrows():
        b = baseline(r["series"])
        e = r["own_elasticity"]
        p_star = optimal_price(b.price, b.units, b.cost, e, bound)
        m0 = margin_curve(np.array([b.price]), b.price, b.units, b.cost, e)[0]
        m1 = margin_curve(np.array([p_star]), b.price, b.units, b.cost, e)[0]
        lo = optimal_price(b.price, b.units, b.cost, r["ci_low"], bound)
        hi = optimal_price(b.price, b.units, b.cost, r["ci_high"], bound)
        unconstrained = b.cost * e / (1 + e) if e < -1 else np.inf
        rows.append({
            "series": r["series"], "elasticity": e, "regular_price": b.price, "unit_cost": b.cost,
            "current_margin_pct": 1 - b.cost / b.price, "optimal_price": p_star,
            "price_change_pct": p_star / b.price - 1, "margin_uplift_pct": m1 / m0 - 1,
            "volume_change_pct": (p_star / b.price) ** e - 1,
            "optimal_price_range_low": min(lo, hi), "optimal_price_range_high": max(lo, hi),
            "unconstrained_optimum": unconstrained,
            "at_bound": abs(p_star / b.price - 1) > bound - 1e-3,
        })
    return pd.DataFrame(rows)


def plot_curve(series: str, opt: pd.DataFrame) -> None:
    r = opt[opt["series"] == series].iloc[0]
    b = baseline(series)
    bound = settings()["scenario"]["price_bounds_pct"]
    p = np.linspace(b.price * (1 - bound), b.price * (1 + bound), 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for e, ls, lab in [(r["elasticity"], "-", f"point estimate (e = {r['elasticity']:.2f})")]:
        m = margin_curve(p, b.price, b.units, b.cost, e)
        ax.plot(p, m, ls=ls, color=plots.ACCENT, label=lab)
    own = pd.read_csv(results("elasticities.csv")).set_index("series").loc[series]
    lo = margin_curve(p, b.price, b.units, b.cost, own["ci_low"])
    hi = margin_curve(p, b.price, b.units, b.cost, own["ci_high"])
    ax.fill_between(p, np.minimum(lo, hi), np.maximum(lo, hi), color=plots.ACCENT, alpha=0.15,
                    label="95% CI of elasticity")
    m0 = margin_curve(np.array([b.price]), b.price, b.units, b.cost, r["elasticity"])[0]
    m1 = margin_curve(np.array([r["optimal_price"]]), b.price, b.units, b.cost, r["elasticity"])[0]
    ax.scatter([b.price], [m0], color=plots.INK, zorder=3, s=40)
    ax.annotate(f"current ${b.price:.2f}", (b.price, m0), xytext=(8, -14), textcoords="offset points",
                fontsize=8.5)
    ax.scatter([r["optimal_price"]], [m1], color=plots.CONTRAST, zorder=3, s=50)
    ax.annotate(f"optimum ${r['optimal_price']:.2f} ({r['margin_uplift_pct']:+.1%} margin)",
                (r["optimal_price"], m1), xytext=(8, 6), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("Regular unit price ($)")
    ax.set_ylabel("Weekly gross margin, all stores ($)")
    ax.set_title(f"{series}: margin vs. price (competitors held fixed)")
    ax.legend(loc="lower center")
    plots.save(fig, "price_optimization_curve.png")


def run(example: str = "Coke | 2L") -> dict:
    opt = optimise_all()
    opt.to_csv(results("price_optimization.csv"), index=False)
    plot_curve(example, opt)
    print(opt[["series", "elasticity", "regular_price", "unit_cost", "current_margin_pct",
               "optimal_price", "price_change_pct", "margin_uplift_pct", "at_bound"]].round(3)
          .to_string(index=False))
    ex = opt[opt["series"] == example].iloc[0]
    return {"example_series": example, "example_price_change_pct": float(ex["price_change_pct"]),
            "example_margin_uplift_pct": float(ex["margin_uplift_pct"]),
            "example_volume_change_pct": float(ex["volume_change_pct"]),
            "n_series_price_increase": int((opt["price_change_pct"] > 0.005).sum()),
            "n_series_price_decrease": int((opt["price_change_pct"] < -0.005).sum())}


if __name__ == "__main__":
    run()


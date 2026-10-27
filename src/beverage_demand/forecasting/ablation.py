"""Feature ablation: how much does each group of features contribute?

LightGBM (tuned settings) is re-trained on the same 9 backtest folds with one feature group removed
at a time. The increase in WAPE versus the full model is the value of that group.
Results: ``results/ablation.csv`` and the ``ablation_results`` table.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from beverage_demand.config import results, settings
from beverage_demand.db import connect, write_table
from beverage_demand.forecasting.backtest import backtest_origins, load_supervised
from beverage_demand.forecasting.features import (
    CALENDAR,
    KNOWN_IN_ADVANCE,
    STORE_ATTRS,
    feature_columns,
)
from beverage_demand.forecasting.metrics import wape
from beverage_demand.forecasting.models import load_tuned_lgb, train_lgb
from beverage_demand.viz import plots
from beverage_demand.viz.plots import plt


def feature_groups(lags: list[int], seasonal_lag: int) -> dict[str, list[str]]:
    history = [f"lag_{k}" for k in lags] + [f"lag_{seasonal_lag}", "rmean_4", "rmean_13",
                                            "rstd_4", "rstd_13"]
    return {
        "sales history (lags, rolling stats)": history,
        "price & promotion": KNOWN_IN_ADVANCE,
        "calendar & holidays": CALENDAR,
        "store demographics": STORE_ATTRS,
    }


def run() -> pd.DataFrame:
    cfg = settings()
    fc = cfg["forecast"]
    full = feature_columns(fc["lags"], fc["seasonal_lag"])
    _, sup = load_supervised(cfg)
    origins = backtest_origins(fc)
    _, rounds = load_tuned_lgb(full, fc["seed"])

    variants = {"full model": full}
    for name, cols in feature_groups(fc["lags"], fc["seasonal_lag"]).items():
        variants[f"without {name}"] = [f for f in full if f not in cols]

    rows = []
    for name, feats in variants.items():
        t0 = time.time()
        params, _ = load_tuned_lgb(feats, fc["seed"])  # monotone constraints follow the feature list
        ys, ps, promo = [], [], []
        for T in origins:
            model = train_lgb(sup[sup["week"] <= T], feats, params, rounds)
            test = sup[(sup["origin_week"] == T) & (sup["week"] > T)]
            ys.append(test["units"].to_numpy())
            ps.append(model.predict(test[feats]))
            promo.append(test["promo"].to_numpy())
        y, p, pr = np.concatenate(ys), np.concatenate(ps), np.concatenate(promo) == 1
        rows.append({"variant": name, "n_features": len(feats), "wape": wape(y, p),
                     "wape_promo": wape(y[pr], p[pr]), "wape_non_promo": wape(y[~pr], p[~pr])})
        print(f"  {name:<45} WAPE {rows[-1]['wape']:.4f} ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    out["wape_increase_vs_full"] = out["wape"] - out.loc[0, "wape"]
    out.to_csv(results("ablation.csv"), index=False)
    with connect() as con:
        write_table(con, "ablation_results", out)
    plot(out)
    return out


def plot(out: pd.DataFrame) -> None:
    d = out.iloc[1:].sort_values("wape_increase_vs_full")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(d["variant"].str.replace("without ", ""), d["wape_increase_vs_full"],
            color=plots.ACCENT, height=0.55)
    for y, (v, w) in enumerate(zip(d["wape_increase_vs_full"], d["wape"], strict=True)):
        ax.text(max(v, 0), y, f"  {v:+.3f} (WAPE {w:.3f})", va="center", fontsize=8.5)
    ax.set_xlabel(f"Increase in WAPE when the group is removed (full model WAPE {out.loc[0, 'wape']:.3f})")
    ax.set_title("Feature ablation: which inputs the forecast needs most")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=plots.INK_2, lw=0.8)
    ax.set_xlim(min(0, d["wape_increase_vs_full"].min() * 1.3), d["wape_increase_vs_full"].max() * 1.5)
    plots.save(fig, "forecast_ablation.png")


if __name__ == "__main__":
    run()

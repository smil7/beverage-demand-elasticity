"""Forecast charts and SHAP explanation for the backtest results."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from beverage_demand.config import processed, results
from beverage_demand.viz import plots
from beverage_demand.viz.plots import plt

LABELS = {"lightgbm": "LightGBM", "regression": "Log-log regression",
          "seasonal_naive": "Seasonal naive", "moving_avg_4": "4-week moving average"}
MODEL_COLORS = {"lightgbm": plots.CATEGORICAL[0], "regression": plots.CATEGORICAL[1],
                "seasonal_naive": plots.CATEGORICAL[2], "moving_avg_4": plots.CATEGORICAL[3]}

FEATURE_NAMES = {
    "ln_price": "own price (log)", "discount": "discount depth", "rmean_13": "13-wk avg sales",
    "rmean_4": "4-wk avg sales", "lag_1": "sales lag 1", "lag_52": "sales same week last year",
    "price_vs_comp": "price vs competitors", "series_code": "brand x pack", "promo_b": "bonus-buy flag",
    "promo_s": "price-reduction flag", "comp_discount": "competitor discount", "rstd_13": "13-wk sales std",
    "ln_reg_price": "regular price (log)", "h": "forecast horizon",
}


def plot_actual_vs_forecast(pred: pd.DataFrame, series_list: list[str]) -> None:
    fig, axes = plt.subplots(len(series_list), 1, figsize=(10, 3.4 * len(series_list)), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), series_list, strict=True):
        d = pred[pred["series"] == s].groupby("week").agg(
            actual=("units", "sum"), lightgbm=("lightgbm", "sum"),
            seasonal_naive=("seasonal_naive", "sum"), promo_share=("promo", "mean"))
        for w in d.index[d["promo_share"] > 0.5]:
            ax.axvspan(w - 0.5, w + 0.5, color=plots.GRID, alpha=0.6, lw=0)
        ax.plot(d.index, d["actual"], color=plots.INK, lw=2, label="Actual")
        ax.plot(d.index, d["lightgbm"], color=MODEL_COLORS["lightgbm"], lw=2, label="LightGBM")
        ax.plot(d.index, d["seasonal_naive"], color=MODEL_COLORS["seasonal_naive"], lw=1.5,
                ls="--", label="Seasonal naive")
        ax.set_title(f"{s}: weekly units, all stores (shaded = most stores on promotion)")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    np.atleast_1d(axes)[0].legend(loc="upper left", ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel("Dominick's week (backtest period, 1-8 weeks ahead forecasts)")
    fig.tight_layout()
    plots.save(fig, "forecast_actual_vs_forecast.png")


def plot_wape_by_series(m: pd.DataFrame) -> None:
    d = m[m["slice"] == "series"].pivot(index="group", columns="model", values="wape")
    d["best_baseline"] = d[["seasonal_naive", "moving_avg_4", "regression"]].min(axis=1)
    d = d.sort_values("lightgbm")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["lightgbm"], height=0.4, color=MODEL_COLORS["lightgbm"], label="LightGBM")
    ax.barh(y - 0.2, d["best_baseline"], height=0.4, color=plots.NEUTRAL, label="Best baseline")
    for yi, v in zip(y, d["lightgbm"], strict=True):
        ax.text(v + 0.01, yi + 0.2, f"{v:.0%}", va="center", fontsize=8)
    ax.set_yticks(y, d.index)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("WAPE (lower is better)")
    ax.set_title("Forecast error by brand x pack (store level, 1-8 weeks ahead)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    plots.save(fig, "forecast_wape_by_series.png")


def plot_promo_split(m: pd.DataFrame) -> None:
    d = m[m["slice"] == "promo"].pivot(index="model", columns="group", values="wape")
    order = ["lightgbm", "regression", "seasonal_naive", "moving_avg_4"]
    d = d.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(2)
    w = 0.2
    for i, mdl in enumerate(order):
        vals = d.loc[mdl, ["non-promo weeks", "promo weeks"]].to_numpy(float)
        ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=MODEL_COLORS[mdl], label=LABELS[mdl],
               edgecolor=plots.SURFACE, linewidth=1.5)
        for xi, v in zip(x + (i - 1.5) * w, vals, strict=True):
            ax.text(xi, min(v, 1.95) + 0.03, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Non-promo weeks", "Promo weeks"])
    ax.set_ylim(0, 2.1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("WAPE (bars capped at 200%)")
    ax.set_title("LightGBM has the lowest error in both promo and non-promo weeks")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=1, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    plots.save(fig, "forecast_wape_promo_split.png")


def plot_shap() -> list[tuple[str, float]]:
    model = lgb.Booster(model_file=str(processed("lightgbm_last_fold").with_suffix(".txt")))
    X = pd.read_parquet(processed("shap_sample"))
    sv = shap.TreeExplainer(model).shap_values(X)
    Xn = X.rename(columns=FEATURE_NAMES)
    plt.figure()
    shap.summary_plot(sv, Xn, max_display=15, show=False, plot_size=(8, 6))
    fig = plt.gcf()
    fig.axes[0].set_title("What drives the forecast (SHAP, last backtest fold)", loc="left")
    plots.save(fig, "forecast_shap_summary.png")
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xn.columns).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(results("forecast_shap_importance.csv"))
    return list(imp.head(8).round(4).items())


def plot_horizon(m: pd.DataFrame) -> None:
    d = m[m["slice"] == "horizon"].pivot(index="group", columns="model", values="wape")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for mdl in ["lightgbm", "regression", "seasonal_naive"]:
        ax.plot(d.index.astype(int), d[mdl], marker="o", color=MODEL_COLORS[mdl], label=LABELS[mdl])
    ax.set_xlabel("Weeks ahead")
    ax.set_ylabel("WAPE")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Forecast error by horizon")
    ax.legend()
    plots.save(fig, "forecast_wape_by_horizon.png")


def run() -> dict:
    pred = pd.read_parquet(processed("backtest_predictions"))
    m = pd.read_csv(results("forecast_metrics.csv"))
    plot_actual_vs_forecast(pred, ["Pepsi | 24pk can", "Coke | 2L"])
    plot_wape_by_series(m)
    plot_promo_split(m)
    plot_horizon(m)
    top = plot_shap()
    print("Top SHAP features:", top)
    return {"top_shap_features": [f for f, _ in top]}


if __name__ == "__main__":
    run()

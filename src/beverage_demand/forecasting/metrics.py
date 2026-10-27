"""Forecast accuracy metrics.

WAPE (weighted absolute percentage error) is the primary metric: sum|error| / sum(actual). Unlike
MAPE it is not blown up by weeks with tiny actuals and it weights big-volume items more, which is
what a supply/commercial planner cares about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute percentage error over weeks with positive actuals."""
    m = y > 0
    return float(np.mean(np.abs(y[m] - yhat[m]) / y[m]))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed: positive = over-forecasting, as a share of actual volume."""
    return float((yhat - y).sum() / y.sum())


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def score(df: pd.DataFrame, models: list[str], by: list[str] | None = None,
          actual: str = "units") -> pd.DataFrame:
    """Metrics for each model column, optionally per group. Returns a long table."""
    groups = df.groupby(by) if by else [((), df)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        y = g[actual].to_numpy(float)
        for m in models:
            p = g[m].to_numpy(float)
            row = dict(zip(by or [], key, strict=True))
            row |= {"model": m, "n": len(g), "wape": wape(y, p), "mape": mape(y, p),
                    "bias": bias(y, p), "rmse": rmse(y, p)}
            rows.append(row)
    return pd.DataFrame(rows)

"""Calendar dimension: Dominick's WEEK number -> dates, holidays and seasonality.

Week 1 starts Thursday 1989-09-14 (codebook Part 8, "Week's Decode Table"); weeks run Thu-Wed.
Holiday weeks follow the codebook's "Special Events" column (a week is a holiday week when the
holiday date falls inside it). The codebook does not list the Super Bowl, which matters a lot for
soft drinks, so it is added from the known game dates.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# Super Bowl Sundays 1990-1997 (Super Bowl XXIV - XXXI)
SUPER_BOWL = [
    date(1990, 1, 28), date(1991, 1, 27), date(1992, 1, 26), date(1993, 1, 31),
    date(1994, 1, 30), date(1995, 1, 29), date(1996, 1, 28), date(1997, 1, 26),
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th ``weekday`` (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return date(year, month, day)


def holiday_dates(year: int) -> dict[str, date]:
    return {
        "new_year": date(year, 1, 1),
        "super_bowl": next((d for d in SUPER_BOWL if d.year == year), date(year, 1, 28)),
        "presidents_day": _nth_weekday(year, 2, 0, 3),
        "easter": _easter(year),
        "memorial_day": _nth_weekday(year, 5, 0, -1),
        "july_4": date(year, 7, 4),
        "labor_day": _nth_weekday(year, 9, 0, 1),
        "halloween": date(year, 10, 31),
        "thanksgiving": _nth_weekday(year, 11, 3, 4),
        "christmas": date(year, 12, 25),
    }


HOLIDAYS = list(holiday_dates(1990).keys())


def build_dim_calendar(week1_start: str = "1989-09-14", n_weeks: int = 400) -> pd.DataFrame:
    """One row per Dominick's week with start/end dates, holiday flags and Fourier terms."""
    start = pd.Timestamp(week1_start)
    weeks = np.arange(1, n_weeks + 1)
    cal = pd.DataFrame({"week": weeks})
    cal["week_start"] = pd.date_range(start, periods=n_weeks, freq="7D")
    cal["week_end"] = cal["week_start"] + pd.Timedelta(days=6)
    mid = cal["week_start"] + pd.Timedelta(days=3)
    cal["year"] = mid.dt.year
    cal["month"] = mid.dt.month
    cal["quarter"] = mid.dt.quarter
    cal["week_of_year"] = mid.dt.isocalendar().week.astype(int).clip(upper=52)

    for h in HOLIDAYS:
        flags = []
        for ws, we in zip(cal["week_start"].dt.date, cal["week_end"].dt.date, strict=True):
            hit = any(ws <= holiday_dates(y)[h] <= we for y in {ws.year, we.year})
            flags.append(int(hit))
        cal[f"hol_{h}"] = flags
    hol_cols = [f"hol_{h}" for h in HOLIDAYS]
    cal["is_holiday"] = cal[hol_cols].max(axis=1)
    cal["holiday_name"] = [
        next((h for h in HOLIDAYS if row[f"hol_{h}"]), "") for _, row in cal[hol_cols].iterrows()
    ]
    # annual Fourier terms on day-of-year (smooth seasonality; 52.18 weeks per year)
    doy = mid.dt.dayofyear.to_numpy()
    for k in (1, 2, 3):
        cal[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cal[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return cal

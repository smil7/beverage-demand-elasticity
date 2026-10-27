"""Settings loader. Everything configurable lives in ``config/settings.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    """Return the parsed ``config/settings.yaml``."""
    with open(REPO_ROOT / "config" / "settings.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Absolute path for a key under ``paths:`` in settings (created if missing)."""
    p = REPO_ROOT / settings()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_file(key: str) -> Path:
    return path("raw_dir") / settings()["raw_files"][key]


def processed(name: str) -> Path:
    """Path of a processed Parquet table, e.g. ``processed("fact_sales_weekly")``."""
    return path("processed_dir") / f"{name}.parquet"


def results(name: str) -> Path:
    return path("results_dir") / name


def figure(name: str) -> Path:
    return path("figures_dir") / name

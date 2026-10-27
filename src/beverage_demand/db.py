"""Small helper layer around the DuckDB warehouse (``data/warehouse.duckdb``).

All tables -- raw staging, the star schema, the modelling panel and model outputs such as forecasts
and promotion events -- live in one DuckDB file. SQL lives in ``sql/`` and is run with
``run_sql_file``; ``${name}`` placeholders are filled from keyword arguments.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from beverage_demand.config import REPO_ROOT, settings

SQL_DIR = REPO_ROOT / "sql"


def warehouse_path() -> Path:
    p = REPO_ROOT / settings()["paths"]["warehouse"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the warehouse (or ``":memory:"`` for tests)."""
    target = str(path) if path is not None else str(warehouse_path())
    if target == ":memory:":
        return duckdb.connect(target)
    return duckdb.connect(target, read_only=read_only)


def render(sql: str, **params: object) -> str:
    """Replace ``${name}`` placeholders; fail loudly if one is missing."""
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in params:
            raise KeyError(f"SQL parameter ${{{key}}} not provided")
        return str(params[key])

    return re.sub(r"\$\{(\w+)\}", sub, sql)


def run_sql_file(con: duckdb.DuckDBPyConnection, path: str | Path, **params: object) -> None:
    """Execute every statement of a .sql file (relative paths are resolved under ``sql/``)."""
    path = Path(path)
    if not path.is_absolute():
        path = SQL_DIR / path
    con.execute(render(path.read_text(encoding="utf-8"), **params))


def query(con: duckdb.DuckDBPyConnection, sql: str, **params: object) -> pd.DataFrame:
    return con.execute(render(sql, **params)).df()


def query_file(con: duckdb.DuckDBPyConnection, path: str | Path, **params: object) -> pd.DataFrame:
    path = Path(path)
    if not path.is_absolute():
        path = SQL_DIR / path
    return query(con, path.read_text(encoding="utf-8"), **params)


def table(name: str, columns: list[str] | None = None, where: str | None = None,
          con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Read a warehouse table into pandas."""
    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM {name}" + (f" WHERE {where}" if where else "")
    if con is not None:
        return con.execute(sql).df()
    with connect(read_only=True) as c:
        return c.execute(sql).df()


def write_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Create or replace a table from a pandas DataFrame (used for model outputs)."""
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
    con.unregister("_df")


def sql_list(values: list[str]) -> str:
    """``['Coke', '7Up']`` -> ``'Coke', '7Up'`` for use in an IN (...) clause."""
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)

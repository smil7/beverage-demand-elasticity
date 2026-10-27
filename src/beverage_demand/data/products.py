"""Product hierarchy: map Dominick's UPC descriptions to brand and pack size.

The brand rules live in ``config/brand_mapping.csv`` (ordered regexes, first match wins) so the
mapping is auditable and version controlled. Pack size is parsed from the codebook ``SIZE`` field.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from beverage_demand.config import REPO_ROOT

LITRE_OZ = 33.814
# 1 standard "8-oz equivalent case" = 24 x 8 oz = 192 oz, the usual soft-drink volume unit
EQ_CASE_OZ = 192.0


def load_brand_rules(path: Path | None = None) -> pd.DataFrame:
    path = path or REPO_ROOT / "config" / "brand_mapping.csv"
    return pd.read_csv(path).sort_values("priority").reset_index(drop=True)


def clean_description(desc: str) -> str:
    """Upper-case and strip the codebook's leading status markers (``~``, ``$``, ``#``)."""
    return re.sub(r"^[~$#*]+", "", str(desc).upper()).strip()


def map_brand(desc: str, rules: pd.DataFrame) -> str:
    d = clean_description(desc)
    for pattern, brand in zip(rules["pattern"], rules["brand"], strict=True):
        if re.search(pattern, d):
            return brand
    return "Other"


def parse_size(size: str) -> tuple[int, float]:
    """Parse a codebook SIZE string into (items per pack, ounces per item).

    Examples: ``"24/12O"`` -> (24, 12.0); ``"2 LT"`` -> (1, 67.6); ``"20 OZ"`` -> (1, 20.0).
    Returns (1, nan) when the size cannot be parsed.
    """
    s = str(size).upper().strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)), float(m.group(2))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(LT|LITE|LITER|L\b)", s)
    if m:
        return 1, float(m.group(1)) * LITRE_OZ
    m = re.match(r"^(\d*\.?\d+)\s*(OZ|O\b|O$)", s)
    if m:
        return 1, float(m.group(1))
    return 1, float("nan")


def pack_category(n_items: int, oz_item: float) -> str:
    """Business pack segment used throughout the project."""
    if n_items == 1:
        if abs(oz_item - 2 * LITRE_OZ) < 1:
            return "2L"
        if abs(oz_item - 3 * LITRE_OZ) < 1:
            return "3L"
        if 30 <= oz_item <= 55:
            return "1-1.5L"
        if 7 <= oz_item <= 28:
            return "single serve"
        return "other"
    if abs(oz_item - 12) < 0.5:
        return {6: "6pk can", 12: "12pk can", 24: "24pk can"}.get(n_items, "other can multipack")
    if 7 <= oz_item <= 25:
        return "bottle multipack"
    return "other"


def build_dim_product(upc: pd.DataFrame, rules: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build ``dim_product`` (one row per UPC) from the raw UPC file."""
    rules = load_brand_rules() if rules is None else rules
    out = upc.rename(columns=str.upper).copy()
    out["brand"] = out["DESCRIP"].map(lambda d: map_brand(d, rules))
    parsed = out["SIZE"].map(parse_size)
    out["items_per_pack"] = [p[0] for p in parsed]
    out["oz_per_item"] = [p[1] for p in parsed]
    out["oz_per_unit"] = out["items_per_pack"] * out["oz_per_item"]
    out["pack"] = [pack_category(n, oz) for n, oz in parsed]
    out["description"] = out["DESCRIP"].map(clean_description)
    out = out.rename(columns={"UPC": "upc", "SIZE": "size", "CASE": "case_pack", "COM_CODE": "com_code"})
    cols = ["upc", "description", "size", "case_pack", "com_code", "brand", "pack",
            "items_per_pack", "oz_per_item", "oz_per_unit"]
    return out[cols].drop_duplicates("upc").reset_index(drop=True)

import pandas as pd
import pytest

from beverage_demand.ingest.calendar import build_dim_calendar
from beverage_demand.ingest.product import (
    build_dim_product,
    load_brand_rules,
    map_brand,
    pack_category,
    parse_size,
)
from beverage_demand.ingest.quality import check_fact

RULES = load_brand_rules()


@pytest.mark.parametrize(
    ("desc", "brand"),
    [
        ("COKE CLASSIC CANS", "Coke"),
        ("COCA-COLA CLASSIC", "Coke"),
        ("CAFFEINE FREE CLASSI", "Coke"),
        ("~DIET COKE CANS", "Diet Coke"),
        ("COKE DIET CAFFEINE F", "Diet Coke"),
        ("PEPSI COLA CANS", "Pepsi"),
        ("PEPSI DIET N/R", "Diet Pepsi"),
        ("WILD CHERRY PEPSI", "Other"),
        ("CRYSTAL DIET PEPSI", "Other"),
        ("SEVEN-UP N/R", "7Up"),
        ("DIET 7-UP", "Diet 7Up"),
        ("SEVEN-UP CHERRY", "Other"),
        ("DR PEPPER SUGAR/FREE", "Diet Dr Pepper"),
        ("DOMINICK'S CASE 12OZ", "Private Label"),
        ("DOM COLA SODA", "Private Label"),
        ("WORLD CLASSIC COLA 6", "Other"),
        ("COKE II", "Other"),
        ("CANADA DRY GINGER AL", "Other"),
    ],
)
def test_brand_mapping(desc, brand):
    assert map_brand(desc, RULES) == brand


@pytest.mark.parametrize(
    ("size", "expected"),
    [("24/12O", "24pk can"), ("12/12O", "12pk can"), ("6/12OZ", "6pk can"), ("2 LT", "2L"),
     ("3 LT", "3L"), ("20 OZ", "single serve"), ("8/20OZ", "bottle multipack")],
)
def test_pack_parsing(size, expected):
    assert pack_category(*parse_size(size)) == expected


def test_dim_product_volume():
    upc = pd.DataFrame({"COM_CODE": [1, 1], "UPC": [10, 11], "DESCRIP": ["PEPSI COLA CANS", "DIET COKE"],
                        "SIZE": ["24/12O", "2 LT"], "CASE": [1, 8], "NITEM": [0, 0]})
    d = build_dim_product(upc, RULES)
    assert d.loc[0, "oz_per_unit"] == 288
    assert d.loc[1, "pack"] == "2L"


def test_calendar_matches_codebook():
    cal = build_dim_calendar().set_index("week")
    assert cal.loc[1, "week_start"] == pd.Timestamp("1989-09-14")
    assert cal.loc[400, "week_start"] == pd.Timestamp("1997-05-08")  # codebook: week 400 = 05/08/97
    assert cal.loc[42, "holiday_name"] == "july_4"      # codebook: week 42 = 4th of July 1990
    assert cal.loc[63, "holiday_name"] == "thanksgiving"
    assert cal.loc[103, "holiday_name"] == "labor_day"
    assert cal["is_holiday"].sum() > 60


def test_quality_checks_flag_duplicates():
    cal = build_dim_calendar(n_weeks=5)
    f = pd.DataFrame({"store": [1, 1], "brand": ["Coke"] * 2, "pack": ["2L"] * 2, "week": [1, 1],
                      "units": [5, 6], "unit_price": [1.0, 1.0], "dollars": [5.0, 6.0],
                      "cogs": [4.0, 5.0]})
    checks = {c.name: c for c in check_fact(f, cal)}
    assert not checks["No duplicate brand x pack x store x week keys"].passed
    assert checks["No negative volumes"].passed

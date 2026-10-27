import pandas as pd
import pytest

from beverage_demand.data.calendar import build_dim_calendar
from beverage_demand.data.load import aggregate_weekly, clean_movement
from beverage_demand.data.products import (
    build_dim_product,
    load_brand_rules,
    map_brand,
    pack_category,
    parse_size,
)

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


def test_cleaning_and_weekly_aggregation_follow_the_codebook():
    raw = pd.DataFrame({
        "STORE": [1, 1, 1, 1, 1], "UPC": [10, 10, 10, 10, 10], "WEEK": [1, 2, 3, 4, 5],
        "MOVE": [6, 4, 5, 0, 2], "QTY": [3, 1, 1, 1, 1],
        "PRICE": [2.97, 1.50, 1.50, 0.0, 1.60],     # week 1: 3 for $2.97 -> unit price 0.99
        "SALE": ["B", "", "", "", "S"],
        "PROFIT": [25.0, 20.0, 20.0, 0.0, 999.0],   # week 5: impossible margin -> cost unknown
        "OK": [1, 1, 0, 1, 1],                      # week 3: flagged as suspect -> dropped
    })
    clean = clean_movement(raw)
    assert list(clean["week"]) == [1, 2, 5]  # OK = 0 and PRICE = 0 rows are gone
    products = pd.DataFrame({"upc": [10], "brand": ["Coke"], "pack": ["2L"], "oz_per_unit": [67.6]})
    f = aggregate_weekly(clean, products).set_index("week")
    assert f.loc[1, "unit_price"] == pytest.approx(0.99)
    assert f.loc[1, "dollars"] == pytest.approx(2.97 * 6 / 3)
    assert f.loc[1, "cogs"] == pytest.approx(2.97 * 6 / 3 * 0.75)
    assert f.loc[1, "units_sale_b"] == 6 and f.loc[2, "any_sale_b"] == 0
    assert f.loc[5, "any_sale_s"] == 1
    assert pd.isna(f.loc[5, "cogs"])

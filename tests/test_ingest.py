import pandas as pd
import pytest

from beverage_demand.db import run_sql_file
from beverage_demand.ingest.calendar import build_dim_calendar
from beverage_demand.ingest.product import (
    build_dim_product,
    load_brand_rules,
    map_brand,
    pack_category,
    parse_size,
)
from beverage_demand.ingest.quality import run_checks

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


def test_warehouse_sql_applies_codebook_rules(tmp_path):
    """Staging -> movement_clean -> fact_sales_weekly on a 5-row CSV."""
    from sqlutil import db_with_fact

    csv = tmp_path / "wsdr.csv"
    lines = [
        "STORE,UPC,WEEK,MOVE,QTY,PRICE,SALE,PROFIT,OK,PRICE_HEX,PROFIT_HEX",
        "1,1,1,6,3,2.97,B,25,1,x,x",    # bundle: 3 for $2.97 -> unit price 0.99
        "1,1,2,4,1,1.50,,20,1,x,x",
        "1,1,3,5,1,1.50,,20,0,x,x",     # OK = 0 -> dropped
        "1,1,4,0,1,0,,0,1,x,x",         # PRICE = 0 (not stocked) -> dropped
        "1,1,5,2,1,1.60,S,999,1,x,x",   # implausible margin -> cost unknown
    ]
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    con = db_with_fact([])
    con.execute("DROP VIEW movement_clean")
    con.execute("DROP TABLE stg_movement")
    run_sql_file(con, "warehouse/02_staging.sql", movement_csv=csv.as_posix())
    run_sql_file(con, "warehouse/03_movement_clean.sql")
    run_sql_file(con, "warehouse/04_fact_sales_weekly.sql")
    f = con.execute("SELECT * FROM fact_sales_weekly ORDER BY week").df().set_index("week")
    assert list(f.index) == [1, 2, 5]
    assert f.loc[1, "unit_price"] == pytest.approx(0.99)
    assert f.loc[1, "dollars"] == pytest.approx(2.97 * 6 / 3)
    assert f.loc[1, "cogs"] == pytest.approx(2.97 * 6 / 3 * 0.75)
    assert f.loc[1, "units_sale_b"] == 6 and f.loc[2, "any_sale_b"] == 0
    assert pd.isna(f.loc[5, "cogs"])


def test_primary_key_rejects_duplicates():
    import duckdb
    from sqlutil import db_with_fact, fact_rows

    con = db_with_fact(fact_rows(1, "Coke", "2L", [1], [2.0]))
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO fact_sales_weekly SELECT * FROM fact_sales_weekly")


def test_quality_checks_pass_and_warn():
    from sqlutil import db_with_fact, fact_rows

    rows = fact_rows(1, "Coke", "2L", [1, 2], [2.0, 30.0]) + fact_rows(3, "Coke", "2L", [1], [2.0])
    con = db_with_fact(rows)
    checks = run_checks(con).set_index("check_name")
    assert checks.loc["No negative volumes", "status"] == "PASS"
    assert checks.loc["Unit price sane (<= $25 per pack)", "status"] == "WARN"  # the $30 row
    assert checks.loc["Every store is in dim_store", "failing_rows"] == 1      # store 3

# Data

This project uses the **Dominick's Finer Foods** scanner data (Soft Drinks category) from the
[James M. Kilts Center for Marketing, University of Chicago Booth School of Business](https://www.chicagobooth.edu/research/kilts/research-data/dominicks).

> The data are provided **for academic research purposes only**, and users must acknowledge the
> Kilts Center for Marketing at the University of Chicago Booth School of Business. For that reason
> **no raw or processed data are committed to this repository** (`data/` is git-ignored). Only small,
> aggregated model outputs (elasticity tables, metric summaries, charts) live in `results/`.

## Automatic download

```bash
uv run python scripts/download_data.py
```

This fetches the files below into `data/raw/` and unzips them (~100 MB download, ~1.3 GB on disk).

## Manual download (if the script fails)

1. Open https://www.chicagobooth.edu/research/kilts/research-data/dominicks
2. Under **Category-specific files -> Soft Drinks**, download:
   - Movement file, CSV version: `wsdr.zip` -> unzip to `data/raw/wsdr.csv`
   - UPC file, CSV version: `upcsdr.csv` -> `data/raw/upcsdr.csv`
3. Under **Store-specific demographics**, download the Stata version `demo_stata.zip`
   -> unzip to `data/raw/demo.dta`
4. Download the **Manual and Codebook** PDF to
   `data/raw/dominicks-manual-and-codebook_kiltscenter.pdf` (optional, for reference).

Expected layout:

```
data/raw/
  wsdr.csv          # 17,730,501 rows: STORE, UPC, WEEK, MOVE, QTY, PRICE, SALE, PROFIT, OK, PRICE_HEX, PROFIT_HEX
  upcsdr.csv        # 1,746 UPCs: COM_CODE, UPC, DESCRIP, SIZE, CASE, NITEM
  demo.dta          # store demographics incl. zone and price tier (priclow/pricmed/prichigh)
```

## Field definitions used (verified against the codebook)

| Field | Meaning | How it is used |
|---|---|---|
| `MOVE` | number of items sold | units |
| `QTY`, `PRICE` | price `PRICE` is for a bundle of `QTY` items | unit price = `PRICE / QTY`; dollars = `PRICE * MOVE / QTY` |
| `PROFIT` | retailer gross margin, % of price (average acquisition cost basis) | cost = dollars x (1 - PROFIT/100) |
| `SALE` | B = bonus buy, C = coupon, S = simple price reduction; `G` appears but is undocumented | promotion type |
| `OK` | 0 = suspect data | rows with `OK = 0` are dropped |
| `WEEK` | week 1 = 1989-09-14 (Thu) ... week 400 = 1997-05-08 | calendar dimension |

Rows with `OK = 0`, `PRICE <= 0` or `MOVE < 0` are dropped (see `src/beverage_demand/data/load.py`).

## Processed files (generated, also git-ignored)

`python -m beverage_demand.data.load` reads the raw files with pandas and writes to
`data/processed/`: `sales_weekly.parquet` (brand x pack x store x week), `brand_weekly.parquet`
(chain level), `products`, `stores` and `calendar`. The later steps add `model_panel.parquet`,
the backtest predictions and the promotion events.

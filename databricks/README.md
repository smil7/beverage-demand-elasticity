# Running the pipeline on Databricks (Free Edition)

The notebooks in this folder are Databricks *source* notebooks (`# Databricks notebook source`).
They import the same tested code in `src/` and run the same SQL in `sql/` as the local pipeline,
and write their outputs as **Delta tables** in Unity Catalog.

| Notebook | What it does | Delta tables written |
|---|---|---|
| `01_ingest.py` | raw CSV/Stata in a Volume -> SQL views -> data model + DQ checks | `dim_product`, `dim_store`, `dim_calendar`, `fact_sales_weekly`, `fact_brand_week` |
| `02_elasticity.py` | modelling panel + log-log own/cross elasticities | `model_panel`, `elasticities`, `cross_elasticities`, `promo_effects` |
| `03_forecast_promo.py` | rolling-origin LightGBM backtest vs. baselines, promo ROI | `forecast_metrics`, `fact_forecast`, `fact_promo_events`, `promo_roi` |

> Status: these notebooks were written against the Databricks APIs but **have not been executed on a
> Databricks workspace by the author of this repo** (no workspace was available while building it).
> Everything they call is exercised locally by `make all` and the test-suite; if a Databricks-specific
> step fails, it will be in the I/O cells (Volume paths, `saveAsTable`), not the modelling code.

## Step by step

1. **Sign up** for [Databricks Free Edition](https://www.databricks.com/learn/free-edition) and open
   the workspace. Free Edition uses serverless compute; nothing to configure.

2. **Clone the repo as a Git folder**: *Workspace -> your user folder -> Create -> Git folder*, URL
   `https://github.com/smil7/beverage-demand-elasticity`. The notebooks find `src/`, `sql/` and
   `config/` relative to themselves, so they must run from inside this Git folder.

3. **Create a schema and a Volume** (SQL editor or a notebook cell):

   ```sql
   CREATE SCHEMA IF NOT EXISTS workspace.beverage;
   CREATE VOLUME IF NOT EXISTS workspace.beverage.raw;
   ```

4. **Upload the raw files** to the Volume: *Catalog -> workspace -> beverage -> raw -> Upload to this
   volume*. Upload `wsdr.csv` (unzipped, ~1.2 GB), `upcsdr.csv` and `demo.dta` -- see
   `data/README.md` for where to download them. (If the browser upload of the 1.2 GB file is
   slow, upload `wsdr.zip` and unzip it from a notebook:
   `import zipfile; zipfile.ZipFile("/Volumes/workspace/beverage/raw/wsdr.zip").extractall("/Volumes/workspace/beverage/raw")`.)

5. **Run the notebooks in order** -- `01_ingest`, `02_elasticity`, `03_forecast_promo`. Each has
   widgets for `catalog` (default `workspace`), `schema` (default `beverage`) and, in 01, the Volume
   path. Notebook 03 has `n_folds` (default 3 on Free Edition; 9 reproduces the local backtest).

6. **Check the outputs** in *Catalog -> workspace -> beverage*; e.g.

   ```sql
   SELECT series, own_elasticity, ci_low, ci_high FROM workspace.beverage.elasticities ORDER BY own_elasticity;
   SELECT * FROM workspace.beverage.forecast_metrics WHERE model = 'lightgbm';
   ```

7. **Optional -- Power BI**: connect Power BI Desktop to the workspace with the *Azure Databricks*
   connector (server hostname + HTTP path from *SQL Warehouses -> Connection details*) and build
   the dashboard in `powerbi/DASHBOARD_SPEC.md` directly on the Delta tables instead of the CSV export.

## Notes

- `%pip install` cells add LightGBM / statsmodels to the serverless environment; the notebook
  restarts Python after installing, which is expected.
- Free Edition has limited compute; 3 folds of the backtest take a few minutes. The local run of
  all 9 folds takes about 3 minutes on a 16-core laptop.
- Natural next steps on Databricks: log each fold to **MLflow** (params, WAPE, the LightGBM model),
  schedule the three notebooks as a **Workflow job**, and register the forecast model in Unity Catalog.

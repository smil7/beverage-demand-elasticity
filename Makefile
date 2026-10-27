.PHONY: download warehouse quality features elasticity tune forecast ablation promo summary notebooks test lint all

RUN = uv run

download:     ## fetch the Dominick's soft-drink files into data/raw (see data/README.md)
	$(RUN) python scripts/download_data.py

warehouse:    ## raw CSV -> DuckDB staging -> star schema (data/warehouse.duckdb)
	$(RUN) bev warehouse

quality:      ## SQL data-quality checks -> results/data_quality_report.md
	$(RUN) bev quality

features:     ## modelling panel built in SQL (sql/features/model_panel.sql)
	$(RUN) bev features

elasticity:   ## log-log price elasticity models
	$(RUN) bev elasticity

tune:         ## LightGBM hyperparameter search with time-series CV
	$(RUN) bev tune

forecast:     ## rolling-origin backtest: LightGBM, HistGradientBoosting, 3 baselines, SHAP
	$(RUN) bev forecast

ablation:     ## feature-group ablation study
	$(RUN) bev ablation

promo:        ## promotion lift and ROI (SQL on the forecast outputs)
	$(RUN) bev promo

summary:      ## results/summary.json
	$(RUN) bev summary

notebooks:    ## re-execute the notebooks in place
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests scripts

all: warehouse quality features elasticity tune forecast ablation promo summary notebooks

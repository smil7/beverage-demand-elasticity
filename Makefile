.PHONY: download data features elasticity forecast promo scenario powerbi summary notebooks test lint all

RUN = uv run

download:            ## fetch Dominick's soft-drink files into data/raw (see data/README.md)
	$(RUN) python scripts/download_data.py

data: download       ## PySpark ingestion + SQL views -> Parquet, data-quality report
	$(RUN) python -m beverage_demand.ingest.spark_pipeline
	$(RUN) python -m beverage_demand.ingest.quality

features:            ## modelling panel (regular price, discount, promo types, competitor prices)
	$(RUN) python -m beverage_demand.features.panel

elasticity:          ## Phase 3: own/cross elasticities, robustness, heatmap
	$(RUN) bev elasticity

forecast:            ## Phase 4: rolling-origin backtest, baselines, SHAP
	$(RUN) bev forecast

promo:               ## Phase 5: promo lift, ROI ranking (needs forecast)
	$(RUN) bev promo

scenario:            ## Phase 6: price optimisation + example scenario
	$(RUN) bev scenario

powerbi:             ## Phase 7: star-schema export to powerbi/export
	$(RUN) bev powerbi

summary:             ## results/summary.json (README numbers)
	$(RUN) bev summary

notebooks:           ## re-execute the narrative notebooks in place
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests scripts

all: data features elasticity forecast promo scenario powerbi summary notebooks

PYTHON ?= python

.PHONY: install format lint test run-api download-data prepare-data run-popularity run-itemknn run-als run-als-experiments build-ranking-dataset run-ranker run-experiments

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

format:
	$(PYTHON) -m black src scripts tests

lint:
	$(PYTHON) -m ruff check src scripts tests

test:
	$(PYTHON) -m pytest

run-api:
	$(PYTHON) -m uvicorn src.serving.app:app --reload

download-data:
	$(PYTHON) -m scripts.download_data

prepare-data:
	$(PYTHON) -m scripts.prepare_data

run-popularity:
	$(PYTHON) -m scripts.run_popularity_baseline

run-itemknn:
	$(PYTHON) -m scripts.run_itemknn_baseline

run-als:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m scripts.run_als_baseline

run-als-experiments:
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m scripts.run_als_experiments

build-ranking-dataset:
	$(PYTHON) -m scripts.build_ranking_dataset --mode train

build-ranking-test:
	$(PYTHON) -m scripts.build_ranking_dataset --mode test

run-ranker-valid:
	$(PYTHON) -m scripts.run_lightgbm_ranker --mode valid

run-ranker-test:
	$(PYTHON) -m scripts.run_lightgbm_ranker --mode test

run-experiments:
	OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1  $(PYTHON) -m scripts.run_experiment

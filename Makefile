PYTHON ?= python

.PHONY: install format lint test run-api

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

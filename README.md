# Two-Stage Recommendation System

A production-style recommender system implementing candidate retrieval and learning-to-rank pipelines using implicit feedback data.

## Overview

This repository provides the initial scaffold for a two-stage recommendation system designed with machine learning engineering best practices in mind. The long-term goal is to support reproducible training, offline evaluation, and online serving for implicit feedback recommendation use cases.

## System Architecture

The system is organized into two primary stages:

### Candidate Generation Stage

The first stage retrieves a manageable set of relevant candidate items for each user. Planned approaches include popularity baselines, item-based nearest neighbors, and Alternating Least Squares (ALS) models trained on implicit interaction data.

### Ranking Stage

The second stage scores and reorders the retrieved candidates using a learning-to-rank model. The repository is structured to support feature generation, ranking dataset construction, LightGBM training, and batch or online prediction.

## Tech Stack

- Python
- pandas
- numpy
- scipy
- implicit
- LightGBM
- FastAPI
- pytest

## Project Roadmap

- dataset ingestion
- preprocessing pipeline
- candidate generation
- ranking model
- evaluation metrics
- API serving

## Repository Layout

Key directories:

- `configs/` for model and pipeline configuration
- `scripts/` for training and evaluation entrypoints
- `src/` for reusable application code
- `tests/` for unit tests
- `data/` for local datasets
- `artifacts/` for generated models, features, and reports

## Quickstart

1. Create a virtual environment.
2. Install dependencies with `make install`.
3. Copy `.env.example` to `.env` if needed.
4. Run tests with `make test`.
5. Start the API with `make run-api`.

## Kaggle API Setup

Kaggle credentials must **not** be stored in this repository. Do not commit `kaggle.json` or any real API keys.

Generate your API token from:
https://www.kaggle.com/settings/account

Then install it in your local Kaggle configuration directory:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Use [configs/kaggle.example.json](configs/kaggle.example.json) only as a format reference. Keep real credentials outside version control.

## Data Preparation

The current data pipeline targets the Kaggle dataset `retailrocket/ecommerce-dataset`.

Kaggle credentials must be configured locally before downloading data. Place your token at `~/.kaggle/kaggle.json` and set secure permissions:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Run the raw download and preprocessing pipeline with:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

Processed artifacts are written under `data/processed/`:

- `interactions.parquet`
- `train.parquet`
- `val.parquet`
- `test.parquet`
- `item_features.parquet`

Interaction preprocessing standardizes the RetailRocket schema to `user_id`, `item_id`, `event_type`, `timestamp`, and `event_weight`. Event weights follow an implicit-feedback heuristic: `view -> 1.0`, `addtocart -> 3.0`, and `transaction -> 5.0`. Chronological train, validation, and test splits are created by sorting interactions by time and splitting by row order.

Troubleshooting:

- If `kaggle` is missing, install it with `pip install kaggle` and confirm the CLI is on your `PATH`.
- If `kaggle.json` is missing, create a token from your Kaggle account settings and place it under `~/.kaggle/kaggle.json`.
- If parquet writes fail, install a parquet engine such as `pyarrow`.

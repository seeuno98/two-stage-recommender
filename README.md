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

or 
```bash
make download-data
make prepare-data
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

## Baseline Recommendation

The first working recommender in this project is a popularity baseline. Item scores are computed from `train.parquet` using summed `event_weight`, and users are only recommended items they have not already seen in the training data.

Validation is performed on `val.parquet`, where each user's interacted items are treated as relevant items. Offline evaluation reports `Recall@10`, `Recall@20`, `Recall@50`, `NDCG@10`, `NDCG@20`, and `NDCG@50`.

Run the baseline with:

```bash
python -m scripts.run_popularity_baseline
```

This baseline matters because it provides a simple benchmark before moving to ALS candidate generation and more advanced retrieval models.

## Personalized Candidate Generation Baseline

This step introduces the first personalized recommender in the project. Recommendations are based on item-item co-occurrence across user histories in `train.parquet`, so each user's candidates depend on the items they previously interacted with.

Seen items are filtered from the recommendation list, and validation is performed on `val.parquet` using `Recall@10`, `Recall@20`, `Recall@50`, `NDCG@10`, `NDCG@20`, and `NDCG@50`.

Run the personalized baseline with:

```bash
python -m scripts.run_itemknn_baseline
```

This matters because it provides a stronger personalized benchmark before moving to ALS candidate generation.

## ALS Candidate Generation Baseline

This step introduces matrix-factorization-based candidate retrieval. ALS is trained on implicit feedback using weighted user-item interactions from `train.parquet`, with `event_weight` used as the interaction strength and confidence signal.

The implicit ALS model is fit on a user-item CSR matrix, and recommendation uses the matching user row from that same user-item matrix so returned indices decode correctly back into original item IDs. Unseen users in validation return no recommendations.

Seen items are filtered from recommendations, and validation is performed on `val.parquet` using `Recall@10`, `Recall@20`, `Recall@50`, `NDCG@10`, `NDCG@20`, and `NDCG@50`.

Run the ALS baseline with:

```bash
python -m scripts.run_als_baseline
```

This matters because it is the first latent-factor personalized retrieval model and should be a stronger benchmark before ranking.

If `implicit` installation is problematic in your environment, use Python 3.11 for best compatibility.
If BLAS oversubscription warnings appear, run with `OPENBLAS_NUM_THREADS=1`.

## Result Summary
* A naive item-item co-occurrence baseline underperformed the popularity baseline on RetailRocket, suggesting that raw co-occurrence over sparse/noisy implicit events was not sufficient for strong candidate retrieval without additional normalization or stronger-signal filtering.

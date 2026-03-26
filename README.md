# Two-Stage Recommendation System

A production-style recommender system implementing candidate retrieval and learning-to-rank pipelines using implicit feedback data.

## Overview

This repository implements an end-to-end two-stage recommender system with candidate retrieval and learning-to-rank, designed for reproducible training, offline evaluation, and extensibility to production systems.

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

## ALS Retrieval Experiments

The first ALS baseline underperformed both the popularity baseline and the item-item baseline on RetailRocket. A likely issue is that weak-intent interactions, especially raw views, add noise to the user-item matrix and hurt retrieval quality.

To diagnose that, the repository includes an ALS experiment runner that compares stronger-signal filtering strategies under the same validation protocol used for earlier baselines:

- all events
- add-to-cart plus transaction
- transaction only

Each run still fits on `train.parquet`, evaluates on `val.parquet`, treats each validation user's interacted items as relevant items, and excludes items already seen in the original training history. The experiment grid also supports different event-weight mappings and a small ALS hyperparameter sweep.

ALS in `implicit` is a collaborative filtering method for implicit feedback. The model is fit on a user-item CSR matrix, and recommendation must use the matching row from that same user-item matrix for the requested user so item indices decode correctly.

Run the experiments with:

```bash
python -m scripts.run_als_experiments
```

or

```bash
make run-als-experiments
```

Reports are written to `artifacts/reports/als_experiments.json`, `artifacts/reports/als_experiments.csv`, and `artifacts/reports/als_best_experiment.json`.

If OpenBLAS oversubscription warnings appear, set `OPENBLAS_NUM_THREADS=1`. The Makefile target does this by default.

## Ranking Dataset Construction

The ranking stage starts by generating retrieval candidates, then labeling those candidates using future user interactions. In the first implementation, popularity retrieval is the default first-stage source because it currently performs best on validation.

For each target user, the pipeline retrieves top-N unseen items from the training history, then labels each `(user_id, item_id)` candidate row with:

- `label = 1` if the item appears in the user's future target interactions
- `label = 0` otherwise

Each user is one ranking group. This is important because learning-to-rank models consume grouped query data rather than treating rows as independent classification examples.

Build the ranking dataset with:

```bash
python -m scripts.build_ranking_dataset
```

or

```bash
make build-ranking-dataset
```

The dataset is saved to `artifacts/features/ranking_train.parquet`, and a summary report is written to `artifacts/reports/ranking_dataset_summary.json`.

## LightGBM Ranker

The second stage uses LightGBM learning-to-rank to rerank the retrieved candidates. The initial implementation uses `lightgbm.LGBMRanker` with the `lambdarank` objective and evaluates reranked outputs with the same top-K metrics used elsewhere in the repository.

Group or query information is required because each user's candidate set is one ranking group. The training script performs a user-level split so candidate rows from the same user do not leak across train and validation.

Run the ranker with:

```bash
python -m scripts.run_lightgbm_ranker
```

or

```bash
make run-ranker
```

This architecture is hybrid once the ranker combines collaborative retrieval signals with aggregate item features and lightweight item metadata features.

## Final System Architecture

The implemented system follows a two-stage recommendation pipeline:

1. Candidate Generation (Retrieval)
   - Popularity baseline (best-performing retrieval model)
   - Item-item co-occurrence (item-KNN)
   - ALS (implicit matrix factorization)

2. Ranking (Learning-to-Rank)
   - LightGBM (LGBMRanker with LambdaRank objective)
   - Ranks top-N retrieved candidates per user

Pipeline:

Train Data → Candidate Generation → Top-N Candidates → Feature Engineering → LightGBM Ranker → Final Recommendations

## Key Results

| Model                     | Recall@10 |
|--------------------------|----------|
| Popularity (retrieval)   | 0.0075   |
| Item-KNN                 | 0.0027   |
| ALS                      | 0.0019   |
| **LightGBM Ranker (final)** | **0.3802** |

The ranking stage improves Recall@10 by approximately **50×** over the best retrieval baseline.

Additional metrics:

- Recall@20: 0.5537
- Recall@50: 0.8471
- NDCG@10: 0.2428

## Result Summary

- Item-item co-occurrence and ALS baselines underperformed the popularity baseline on RetailRocket, indicating that naive collaborative filtering struggled under sparse and noisy implicit feedback.
- ALS performance degraded further when restricting to high-intent signals (cart/purchase), due to loss of user-item connectivity in the interaction graph.
- Introducing a LightGBM learning-to-rank stage dramatically improved performance, achieving a ~50× increase in Recall@10.
- The final system demonstrates the effectiveness of a two-stage hybrid recommendation architecture combining retrieval and ranking.

## Key Insights

- Popularity outperformed ALS and item-KNN in retrieval due to heavy popularity skew in implicit feedback data.
- Filtering to strong signals (cart, purchase) significantly degraded ALS performance due to loss of user-item graph connectivity.
- Ranking is critical: retrieval alone is insufficient, but a learning-to-rank model can effectively combine weak signals into strong recommendations.
- Hybrid features (user behavior, item popularity, interaction history, metadata) are essential for ranking performance.

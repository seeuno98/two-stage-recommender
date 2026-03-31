# Two-Stage Recommendation System

A production-style recommender system implementing candidate retrieval and learning-to-rank pipelines using implicit feedback data.

## Overview

This repository implements an end-to-end two-stage recommender system with candidate retrieval and learning-to-rank, designed for reproducible training, offline evaluation, and extensibility to production systems.

## System Architecture

The system is organized into two primary stages:

### Candidate Generation Stage

The first stage retrieves a manageable set of relevant candidate items for each user. Implemented approaches include popularity baselines, item-item co-occurrence, and Alternating Least Squares (ALS) models trained on implicit interaction data.

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

Build the ranking dataset with the temporal-consistent modes:

```bash
python -m scripts.build_ranking_dataset --mode train   # build train->val ranking dataset
python -m scripts.build_ranking_dataset --mode test    # build (train+val)->test candidate dataset
```

The training ranking dataset is saved to `artifacts/features/ranking_train.parquet` and the test candidate dataset to `artifacts/features/ranking_test.parquet`. Summary reports are written under `artifacts/reports/`.

## LightGBM Ranker

The second stage uses LightGBM learning-to-rank to rerank the retrieved candidates. The initial implementation uses `lightgbm.LGBMRanker` with the `lambdarank` objective and evaluates reranked outputs with the same top-K metrics used elsewhere in the repository.

Group or query information is required because each user's candidate set is one ranking group. The training script performs a user-level split so candidate rows from the same user do not leak across train and validation.

Run the ranker with two evaluation modes:

```bash
python -m scripts.run_lightgbm_ranker --mode valid   # train on ranking_train.parquet and validate
python -m scripts.run_lightgbm_ranker --mode test    # train on ranking_train.parquet, score ranking_test.parquet
```

By default `--mode test` performs the stricter held-out evaluation using a ranker trained on `train->val` and scored on the `(train+val)->test` candidate set.

This architecture is hybrid once the ranker combines collaborative retrieval signals with aggregate item features and lightweight item metadata features.

## Offline Experiment Framework

This stage adds a practical offline experimentation workflow for comparing recommendation variants in a control-vs-treatment style setup. The framework is inspired by A/B testing patterns, but it remains fully offline: users are deterministically assigned to experiment variants with stable hashing, each variant generates recommendations for its own assigned users, and every variant is evaluated with the same offline metrics on held-out data. In the current experiments, the framework surfaced two useful findings: the reranked pipeline underperformed the popularity baseline under stricter temporal test evaluation, and increasing candidate pool size from 100 to 150 improved reranked Recall@10.

The current implementation supports these first-pass pipeline variants:

- `popularity_only`
- `itemknn_only`
- `als_only`
- `popularity_plus_ranker`

Experiment definitions live in `configs/experiments.yaml`. Each experiment declares a deterministic split such as `control: 50` and `treatment: 50`, plus per-variant pipeline settings like `candidate_k` and optional excluded ranker features.

Run all configured experiments with:

```bash
python -m scripts.run_experiment
```

Run one named experiment with:

```bash
python -m scripts.run_experiment --experiment popularity_vs_ranker
```

or

```bash
make run-experiments
```

## FastAPI Recommendation Service

A lightweight online inference layer sits on top of the existing retrieval and ranking pipeline. The service loads recommendation artifacts at startup, serves recommendations through FastAPI, measures per-request latency, emits structured request logs, and gracefully falls back to popularity-based recommendations when reranking artifacts are unavailable.

The API currently supports two serving pipelines:

- `popularity_only`
- `popularity_plus_ranker`

The default pipeline is `popularity_plus_ranker`, with automatic fallback to `popularity_only` if the LightGBM ranker or its feature artifacts are missing. Unseen users are also handled gracefully by returning global popularity recommendations instead of failing.

This is a local production-style serving layer for development, profiling, and debugging. It is not a deployment platform and does not yet include authentication, persistence, or distributed infrastructure.

### Health Check

```bash
curl http://localhost:8000/health
```

### Recommendation Request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "top_k": 10, "pipeline": "popularity_plus_ranker"}'
```

You can also use the convenience GET endpoint:

```bash
curl "http://localhost:8000/recommend/123?top_k=10&pipeline=popularity_plus_ranker"
```

Run the API locally with:

```bash
make run-api
```

Current latency logging is intended for local profiling and debugging. Each request emits one structured log line that includes the user, effective pipeline, top-K, latency, fallback usage, and recommendation count.

## Serving Performance and Observability

The FastAPI service is optimized with startup initialization, in-memory feature lookups, latency breakdown, request middleware, and lightweight candidate caching. Static artifacts are loaded once at startup, user and item feature maps are precomputed for reranked inference, and a bounded in-memory cache reduces repeated candidate generation work.

The service now measures latency across multiple stages:

- `candidate_generation_ms`
- `feature_build_ms`
- `scoring_ms`
- `total_latency_ms`

For a known-user reranked request (`pipeline=popularity_plus_ranker`), the current local service responds in approximately **17 ms**, with the largest share of latency coming from model scoring rather than candidate generation. In one representative request:

- candidate generation: **0.166 ms**
- feature construction: **3.687 ms**
- ranker scoring: **12.092 ms**
- total latency: **17.017 ms**

For unseen users, the service falls back to the popularity pipeline, which is much faster because it bypasses feature construction and model scoring entirely. In one representative unseen-user request, total latency was approximately **0.046 ms**. :contentReference[oaicite:0]{index=0}

Structured request logs and response metadata include:
- request ID
- requested pipeline
- effective pipeline
- candidate pool size
- whether the user was known
- whether fallback was used
- fallback reason
- whether the ranker actually ran
- latency breakdown
- model version and artifact source paths

This makes the service easier to debug and profile, and helps explain the trade-off between:
- **fast heuristic recommendations** (`popularity_only`)
- **higher-quality but more expensive reranked recommendations** (`popularity_plus_ranker`)

The current service is optimized for **local development, debugging, and profiling**, rather than distributed deployment. Startup loading is implemented with FastAPI lifespan, and request IDs plus timing headers are added with request middleware.

Example request:

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "top_k": 10, "pipeline": "popularity_plus_ranker"}' | python -m json.tool
```

Reports are saved under `artifacts/reports/experiments/<experiment_name>/` and include:

- `results.json`
- `variant_metrics.csv`
- `summary.txt`

Each report includes per-variant user counts, Recall@K / NDCG@K metrics, and simple lift calculations versus the control variant.

This is intentionally an offline framework. It does not simulate live traffic routing, delayed feedback, interference effects, or true online causal impact. Its purpose is to make variant comparison reproducible and operationally similar to control-vs-treatment evaluation without pretending to be a real production experiment platform.

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
## Makefile Commands

The repository includes a Makefile to simplify common workflows for data preparation, model training, evaluation, and experiments.

### Setup

```bash
make install
make test
```

### Data Pipeline
```bash
make download-data
make prepare-data
```

### Retrieval Baselines
```bash
make run-popularity
make run-itemknn
make run-als
make run-als-experiments
```

### Ranking Pipeline
```bash
make build-ranking-train   # build train -> val ranking dataset
make build-ranking-test      # build (train + val) -> test candidate dataset
make run-ranker-valid        # validation-style ranker evaluation
make run-ranker-test         # stricter held-out temporal test evaluation
```

### Offline Experiments
```bash
make run-experiments
```

### API Serving
```bash
make run-api
```

## Reproducing the Main Results

### 1. Prepare the dataset
```bash
make download-data
make prepare-data
```

### 2. Run retrieval baselines
```bash
make run-popularity
make run-itemknn
make run-als
make run-als-experiments
```

### 3. Build ranking datasets
```bash
make build-ranking-dataset
make build-ranking-test
```

### 4. Run ranker evaluations
```bash
make run-ranker-valid
make run-ranker-test
```

### 5. Run offline control-vs-treatment experiments
```bash
make run-experiments
```

## Notes
run-ranker-valid reports validation-stage ranking performance and is useful for model development, but is optimistic relative to true future-data evaluation.
run-ranker-test performs the stricter temporal evaluation using a ranker trained on train -> val and scored on (train + val) -> test.
run-experiments uses deterministic user assignment for offline control-vs-treatment style comparisons.
ALS-related commands set OPENBLAS_NUM_THREADS=1 to reduce thread oversubscription.
run-experiments sets OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1 to keep experiment runs more stable under WSL.

## Key Results

### Retrieval Baselines (validation)

| Model                   | Recall@10 |
|------------------------|----------:|
| Popularity             | 0.0075    |
| Item-KNN               | 0.0027    |
| ALS                    | 0.0019    |

### Ranker Performance

**Validation (user-level split on ranking_train.parquet)**  
- Recall@10: 0.3802  
- Recall@20: 0.5537  
- Recall@50: 0.8471  
- NDCG@10: 0.2428  

**Held-out temporal test ((train+val) -> test)**  
- Recall@10: 0.1601  
- Recall@20: 0.2621  
- Recall@50: 0.5534  
- NDCG@10: 0.0790  

The validation-stage ranker showed strong gains over retrieval-only baselines, but stricter temporal evaluation on held-out test data revealed a substantial generalization gap.

## Result Summary

- Item-item co-occurrence and ALS baselines underperformed the popularity baseline on RetailRocket, indicating that naive collaborative filtering struggled under sparse and noisy implicit feedback.
- ALS performance degraded further when restricting to high-intent signals (cart/purchase), due to loss of user-item connectivity in the interaction graph.
- The LightGBM ranker achieved large gains in validation, improving Recall@10 from 0.0075 to 0.3802, but held-out temporal test evaluation showed a lower Recall@10 of 0.1601, highlighting a meaningful generalization gap.
- Offline control-vs-treatment experiments showed that increasing candidate pool size from 100 to 150 improved reranked Recall@10, indicating a trade-off between candidate recall and ranking noise.
- The final system demonstrates a realistic two-stage recommendation workflow in which retrieval quality, candidate pool size, and temporal generalization all materially affect ranking performance.

## Key Insights

- Popularity outperformed ALS and item-KNN in retrieval due to heavy popularity skew in implicit feedback data.
- Filtering to strong signals (cart, purchase) significantly degraded ALS performance because it reduced user-item graph connectivity and coverage.
- Validation-stage ranking gains did not fully generalize to held-out temporal test data, highlighting the importance of strict temporal evaluation for recommendation systems.
- Candidate pool size exhibited non-monotonic behavior: increasing top-K from 100 to 120 hurt reranking performance, while increasing to 150 improved Recall@10, indicating a trade-off between candidate recall and ranking noise.
- Hybrid ranking features across user behavior, item popularity, interaction history, and metadata were effective, but feature robustness mattered more under held-out test evaluation than under validation.

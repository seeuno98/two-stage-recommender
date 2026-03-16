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

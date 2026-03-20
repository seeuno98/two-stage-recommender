"""Run the popularity recommender baseline on processed RetailRocket data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.candidate_gen.popularity import PopularityRecommender
from src.eval.metrics import evaluate_user_level


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("artifacts/reports")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
METRICS_PATH = REPORTS_DIR / "popularity_baseline_metrics.json"
K_VALUES = [10, 20, 50]


def build_user_history(df: pd.DataFrame) -> dict[int | str, set[object]]:
    """Build per-user interacted item sets."""

    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def build_ground_truth(df: pd.DataFrame) -> dict[int | str, set[object]]:
    """Build validation ground-truth item sets."""

    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def load_processed_split(path: Path) -> pd.DataFrame:
    """Load a processed parquet split."""

    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found at {path}. Run `python scripts/prepare_data.py` first."
        )
    return pd.read_parquet(path)


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Persist evaluation metrics as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    """Train and evaluate the popularity baseline on validation data."""

    train_df = load_processed_split(TRAIN_PATH)
    val_df = load_processed_split(VAL_PATH)

    print(f"[popularity] train_rows={len(train_df)}")
    print(f"[popularity] val_rows={len(val_df)}")

    train_histories = build_user_history(train_df)
    val_ground_truth = build_ground_truth(val_df)

    print(f"[popularity] validation_users={len(val_ground_truth)}")

    recommender = PopularityRecommender().fit(train_df)
    max_k = max(K_VALUES)
    predictions = recommender.recommend_for_users(train_histories, k=max_k)

    for user_id in val_ground_truth:
        predictions.setdefault(
            user_id,
            recommender.recommend_for_user(
                user_id=user_id,
                user_history=train_histories.get(user_id, set()),
                k=max_k,
            ),
        )

    metrics = evaluate_user_level(
        ground_truth=val_ground_truth,
        predictions=predictions,
        k_values=K_VALUES,
    )

    for k in K_VALUES:
        print(f"[popularity] Recall@{k}={metrics[f'recall@{k}']:.4f}")
        print(f"[popularity] NDCG@{k}={metrics[f'ndcg@{k}']:.4f}")

    save_metrics(metrics, METRICS_PATH)
    print(f"[popularity] metrics_report={METRICS_PATH}")


if __name__ == "__main__":
    main()

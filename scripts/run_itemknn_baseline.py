"""Run the item-item co-occurrence recommender baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.candidate_gen.item_knn import ItemKNNRecommender
from src.eval.metrics import evaluate_user_level


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("artifacts/reports")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
ITEMKNN_METRICS_PATH = REPORTS_DIR / "itemknn_baseline_metrics.json"
POPULARITY_METRICS_PATH = REPORTS_DIR / "popularity_baseline_metrics.json"
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


def maybe_load_metrics(path: Path) -> dict[str, float] | None:
    """Load a metrics JSON file if it exists."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {str(key): float(value) for key, value in payload.items()}


def print_metric_comparison(
    popularity_metrics: dict[str, float],
    itemknn_metrics: dict[str, float],
) -> None:
    """Print a short comparison between popularity and item-kNN."""

    print("[itemknn] comparison_vs_popularity")
    for k in K_VALUES:
        print(
            "[itemknn]"
            f" Recall@{k} popularity={popularity_metrics.get(f'recall@{k}', 0.0):.4f}"
            f" itemknn={itemknn_metrics[f'recall@{k}']:.4f}"
        )
        print(
            "[itemknn]"
            f" NDCG@{k} popularity={popularity_metrics.get(f'ndcg@{k}', 0.0):.4f}"
            f" itemknn={itemknn_metrics[f'ndcg@{k}']:.4f}"
        )


def main() -> None:
    """Train and evaluate the item-item co-occurrence baseline."""

    train_df = load_processed_split(TRAIN_PATH)
    val_df = load_processed_split(VAL_PATH)

    print(f"[itemknn] train_rows={len(train_df)}")
    print(f"[itemknn] val_rows={len(val_df)}")

    train_histories = build_user_history(train_df)
    val_ground_truth = build_ground_truth(val_df)

    print(f"[itemknn] validation_users={len(val_ground_truth)}")
    print(f"[itemknn] num_items={train_df['item_id'].nunique()}")

    recommender = ItemKNNRecommender().fit(train_df)
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
        print(f"[itemknn] Recall@{k}={metrics[f'recall@{k}']:.4f}")
        print(f"[itemknn] NDCG@{k}={metrics[f'ndcg@{k}']:.4f}")

    save_metrics(metrics, ITEMKNN_METRICS_PATH)
    print(f"[itemknn] metrics_report={ITEMKNN_METRICS_PATH}")

    popularity_metrics = maybe_load_metrics(POPULARITY_METRICS_PATH)
    if popularity_metrics is not None:
        print_metric_comparison(popularity_metrics=popularity_metrics, itemknn_metrics=metrics)


if __name__ == "__main__":
    main()

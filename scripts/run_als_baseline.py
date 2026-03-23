"""Run the ALS candidate generation baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from src.candidate_gen.als import ALSCandidateGenerator
from src.eval.metrics import evaluate_user_level


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("artifacts/reports")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
ALS_METRICS_PATH = REPORTS_DIR / "als_baseline_metrics.json"
ALS_MODEL_INFO_PATH = REPORTS_DIR / "als_model_info.json"
POPULARITY_METRICS_PATH = REPORTS_DIR / "popularity_baseline_metrics.json"
ITEMKNN_METRICS_PATH = REPORTS_DIR / "itemknn_baseline_metrics.json"
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


def save_json(payload: dict[str, int | float], output_path: Path) -> None:
    """Persist a dictionary as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def maybe_load_metrics(path: Path) -> dict[str, float] | None:
    """Load a metrics JSON file if it exists."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {str(key): float(value) for key, value in payload.items()}


def print_metric_comparison(
    popularity_metrics: dict[str, float] | None,
    itemknn_metrics: dict[str, float] | None,
    als_metrics: dict[str, float],
) -> None:
    """Print a short metric comparison across baselines."""

    print("[als] comparison")
    for k in K_VALUES:
        popularity_value = popularity_metrics.get(f"recall@{k}", 0.0) if popularity_metrics else 0.0
        itemknn_value = itemknn_metrics.get(f"recall@{k}", 0.0) if itemknn_metrics else 0.0
        print(
            "[als]"
            f" Recall@{k}"
            f" popularity={popularity_value:.4f}"
            f" itemknn={itemknn_value:.4f}"
            f" als={als_metrics[f'recall@{k}']:.4f}"
        )

        popularity_ndcg = popularity_metrics.get(f"ndcg@{k}", 0.0) if popularity_metrics else 0.0
        itemknn_ndcg = itemknn_metrics.get(f"ndcg@{k}", 0.0) if itemknn_metrics else 0.0
        print(
            "[als]"
            f" NDCG@{k}"
            f" popularity={popularity_ndcg:.4f}"
            f" itemknn={itemknn_ndcg:.4f}"
            f" als={als_metrics[f'ndcg@{k}']:.4f}"
        )


def main() -> None:
    """Train and evaluate the ALS baseline on validation data."""

    train_df = load_processed_split(TRAIN_PATH)
    val_df = load_processed_split(VAL_PATH)

    print(f"[als] train_rows={len(train_df)}")
    print(f"[als] val_rows={len(val_df)}")

    train_histories = build_user_history(train_df)
    val_ground_truth = build_ground_truth(val_df)
    eval_histories = {
        user_id: train_histories.get(user_id, set())
        for user_id in val_ground_truth.keys()
    }

    print(f"[als] validation_users={len(val_ground_truth)}")
    print(f"[als] unique_train_users={train_df['user_id'].nunique()}")
    print(f"[als] unique_train_items={train_df['item_id'].nunique()}")

    generator = ALSCandidateGenerator().fit(train_df)
    model_info = generator.get_model_info()
    print(
        "[als] hyperparameters"
        f" factors={model_info['factors']}"
        f" regularization={model_info['regularization']}"
        f" iterations={model_info['iterations']}"
        f" alpha={model_info['alpha']}"
    )

    max_k = max(K_VALUES)
    predictions = generator.recommend_for_users(eval_histories, k=max_k)

    metrics = evaluate_user_level(
        ground_truth=val_ground_truth,
        predictions=predictions,
        k_values=K_VALUES,
    )

    for k in K_VALUES:
        print(f"[als] Recall@{k}={metrics[f'recall@{k}']:.4f}")
        print(f"[als] NDCG@{k}={metrics[f'ndcg@{k}']:.4f}")

    if generator.invalid_recommendation_indices_:
        print(f"[als] skipped_invalid_indices={generator.invalid_recommendation_indices_}")

    save_json(metrics, ALS_METRICS_PATH)
    save_json(model_info, ALS_MODEL_INFO_PATH)
    print(f"[als] metrics_report={ALS_METRICS_PATH}")
    print(f"[als] model_info_report={ALS_MODEL_INFO_PATH}")

    popularity_metrics = maybe_load_metrics(POPULARITY_METRICS_PATH)
    itemknn_metrics = maybe_load_metrics(ITEMKNN_METRICS_PATH)
    if popularity_metrics is not None or itemknn_metrics is not None:
        print_metric_comparison(
            popularity_metrics=popularity_metrics,
            itemknn_metrics=itemknn_metrics,
            als_metrics=metrics,
        )


if __name__ == "__main__":
    main()

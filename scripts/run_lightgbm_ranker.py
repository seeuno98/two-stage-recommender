"""Train and evaluate the LightGBM ranker on the ranking dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_ranking_dataset import (
    ITEM_FEATURES_PATH,
    RANKING_DATASET_PATH,
    TRAIN_PATH,
    VAL_PATH,
    build_ranking_dataset,
    load_parquet,
)
from src.ranking.dataset import build_ground_truth, build_group_array
from src.ranking.features import get_feature_columns
from src.ranking.predict import topk_predictions_from_ranked_df
from src.ranking.train_ranker import (
    evaluate_lgbm_ranker,
    maybe_load_metrics,
    save_feature_importance,
    save_lgbm_metrics,
    save_lgbm_model,
    split_ranking_dataset_by_user,
    train_lgbm_ranker,
)
from src.utils.io import save_json


ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODELS_DIR = ARTIFACTS_DIR / "models"
RANKER_METRICS_PATH = REPORTS_DIR / "lightgbm_ranker_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "lightgbm_feature_importance.csv"
MODEL_PATH = MODELS_DIR / "lightgbm_ranker.txt"
RANKER_SUMMARY_PATH = REPORTS_DIR / "lightgbm_ranker_summary.json"
K_VALUES = [10, 20, 50]
PRIMARY_METRIC = "ndcg@20"
BASELINE_PATHS = {
    "popularity": REPORTS_DIR / "popularity_baseline_metrics.json",
    "itemknn": REPORTS_DIR / "itemknn_baseline_metrics.json",
    "als": REPORTS_DIR / "als_baseline_metrics.json",
    "als_best_experiment": REPORTS_DIR / "als_best_experiment.json",
}


def ensure_ranking_dataset() -> pd.DataFrame:
    """Load the ranking dataset, rebuilding it if needed."""

    if RANKING_DATASET_PATH.exists():
        return pd.read_parquet(RANKING_DATASET_PATH)

    train_df = load_parquet(TRAIN_PATH)
    val_df = load_parquet(VAL_PATH)
    item_features_df = load_parquet(ITEM_FEATURES_PATH)
    ranking_df, _, _ = build_ranking_dataset(
        train_df=train_df,
        val_df=val_df,
        item_features_df=item_features_df,
    )
    RANKING_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_parquet(RANKING_DATASET_PATH, index=False)
    return ranking_df


def extract_metrics(payload: dict[str, object] | None) -> dict[str, float] | None:
    """Normalize metrics payloads from baseline reports."""

    if payload is None:
        return None
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return {
            str(key): float(value)
            for key, value in payload["metrics"].items()
        }
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def print_baseline_comparison(metrics: dict[str, float]) -> None:
    """Print the ranker against existing retrieval baselines if available."""

    print("[ranker] comparison")
    print(f"[ranker] model {PRIMARY_METRIC}={metrics[PRIMARY_METRIC]:.4f}")
    for name, path in BASELINE_PATHS.items():
        baseline_metrics = extract_metrics(maybe_load_metrics(path))
        if baseline_metrics is None:
            continue
        print(
            "[ranker]"
            f" baseline={name}"
            f" Recall@20={baseline_metrics.get('recall@20', 0.0):.4f}"
            f" NDCG@20={baseline_metrics.get('ndcg@20', 0.0):.4f}"
        )


def main() -> None:
    """Train and validate the LightGBM ranking stage."""

    ranking_df = ensure_ranking_dataset()
    feature_cols = get_feature_columns(ranking_df)
    train_df, valid_df = split_ranking_dataset_by_user(ranking_df, valid_frac=0.2, random_state=42)

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    group_train = build_group_array(train_df)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["label"]
    group_valid = build_group_array(valid_df)

    model = train_lgbm_ranker(
        X_train=X_train,
        y_train=y_train,
        group_train=group_train,
        X_valid=X_valid,
        y_valid=y_valid,
        group_valid=group_valid,
    )

    valid_ground_truth = build_ground_truth(valid_df[valid_df["label"] == 1][["user_id", "item_id"]])
    metrics, ranked_valid_df = evaluate_lgbm_ranker(
        model=model,
        candidate_df=valid_df,
        feature_cols=feature_cols,
        ground_truth=valid_ground_truth,
        k_values=K_VALUES,
    )
    predictions = topk_predictions_from_ranked_df(ranked_valid_df, k=max(K_VALUES))

    save_lgbm_metrics(metrics, RANKER_METRICS_PATH)
    feature_importance_df = save_feature_importance(model, feature_cols, FEATURE_IMPORTANCE_PATH)
    save_lgbm_model(model, MODEL_PATH)
    save_json(
        {
            "train_users": int(train_df["user_id"].nunique()),
            "valid_users": int(valid_df["user_id"].nunique()),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "feature_count": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "metrics": metrics,
            "model_path": str(MODEL_PATH),
            "feature_importance_path": str(FEATURE_IMPORTANCE_PATH),
            "num_predictions": int(sum(len(items) for items in predictions.values())),
        },
        RANKER_SUMMARY_PATH,
    )

    print(f"[ranker] train_users={train_df['user_id'].nunique()}")
    print(f"[ranker] valid_users={valid_df['user_id'].nunique()}")
    print(f"[ranker] train_rows={len(train_df)}")
    print(f"[ranker] valid_rows={len(valid_df)}")
    print(f"[ranker] feature_count={len(feature_cols)}")
    for k in K_VALUES:
        print(f"[ranker] Recall@{k}={metrics[f'recall@{k}']:.4f}")
        print(f"[ranker] NDCG@{k}={metrics[f'ndcg@{k}']:.4f}")
    print(f"[ranker] metrics={RANKER_METRICS_PATH}")
    print(f"[ranker] feature_importance={FEATURE_IMPORTANCE_PATH}")
    print(f"[ranker] model={MODEL_PATH}")
    print(f"[ranker] summary={RANKER_SUMMARY_PATH}")
    if not feature_importance_df.empty:
        top_feature = feature_importance_df.iloc[0]
        print(
            "[ranker]"
            f" top_feature={top_feature['feature']}"
            f" gain={float(top_feature['importance_gain']):.2f}"
        )

    print_baseline_comparison(metrics)


if __name__ == "__main__":
    main()

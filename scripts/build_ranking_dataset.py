"""Build a first supervised ranking dataset from popularity retrieval candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.candidate_gen.popularity import PopularityRecommender
from src.ranking.dataset import (
    build_ground_truth,
    build_labeled_ranking_dataframe,
    build_user_histories,
    make_candidate_pool_for_users,
)
from src.ranking.features import add_ranking_features
from src.utils.io import save_json


PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
FEATURES_DIR = ARTIFACTS_DIR / "features"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
ITEM_FEATURES_PATH = PROCESSED_DIR / "item_features.parquet"
RANKING_DATASET_PATH = FEATURES_DIR / "ranking_train.parquet"
SUMMARY_PATH = REPORTS_DIR / "ranking_dataset_summary.json"
DEFAULT_TOP_N = 100


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with a clear error if it is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found at {path}. Run `python -m scripts.prepare_data` first."
        )
    return pd.read_parquet(path)


def build_ranking_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    item_features_df: pd.DataFrame | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[pd.DataFrame, list[str], dict[str, float | int]]:
    """Construct the end-to-end ranking dataset for train-to-validation supervision."""

    train_histories = build_user_histories(train_df)
    val_ground_truth = build_ground_truth(val_df)
    eval_histories = {
        user_id: train_histories.get(user_id, set())
        for user_id in val_ground_truth
    }

    retriever = PopularityRecommender().fit(train_df)
    candidate_df = make_candidate_pool_for_users(
        retriever=retriever,
        user_histories=eval_histories,
        top_n=top_n,
    )
    labeled_df = build_labeled_ranking_dataframe(candidate_df, ground_truth=val_ground_truth)
    ranking_df, feature_cols = add_ranking_features(
        candidates=labeled_df,
        train_df=train_df,
        item_features_df=item_features_df,
    )

    positives = int(ranking_df["label"].sum())
    negatives = int(len(ranking_df) - positives)
    summary: dict[str, float | int] = {
        "users": int(ranking_df["user_id"].nunique()),
        "total_candidate_rows": int(len(ranking_df)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives / len(ranking_df)) if len(ranking_df) else 0.0,
        "average_candidates_per_user": float(
            len(ranking_df) / ranking_df["user_id"].nunique()
        )
        if len(ranking_df)
        else 0.0,
        "feature_count": int(len(feature_cols)),
        "top_n": int(top_n),
    }
    return ranking_df, feature_cols, summary


def main() -> None:
    """Build and save the ranking dataset."""

    train_df = load_parquet(TRAIN_PATH)
    val_df = load_parquet(VAL_PATH)
    item_features_df = load_parquet(ITEM_FEATURES_PATH)

    ranking_df, feature_cols, summary = build_ranking_dataset(
        train_df=train_df,
        val_df=val_df,
        item_features_df=item_features_df,
    )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ranking_df.to_parquet(RANKING_DATASET_PATH, index=False)
    save_json(
        {
            **summary,
            "feature_columns": feature_cols,
            "dataset_path": str(RANKING_DATASET_PATH),
        },
        SUMMARY_PATH,
    )

    print(f"[ranking-dataset] users={summary['users']}")
    print(f"[ranking-dataset] rows={summary['total_candidate_rows']}")
    print(f"[ranking-dataset] positives={summary['positives']}")
    print(f"[ranking-dataset] negatives={summary['negatives']}")
    print(f"[ranking-dataset] positive_rate={summary['positive_rate']:.4f}")
    print(
        "[ranking-dataset]"
        f" avg_candidates_per_user={summary['average_candidates_per_user']:.2f}"
    )
    print(f"[ranking-dataset] feature_count={summary['feature_count']}")
    print(f"[ranking-dataset] dataset={RANKING_DATASET_PATH}")
    print(f"[ranking-dataset] summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()

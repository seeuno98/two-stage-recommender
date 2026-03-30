"""Build a first supervised ranking dataset from popularity retrieval candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ranking.dataset import build_ranking_dataset_from_splits
from src.utils.io import save_json


PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
FEATURES_DIR = ARTIFACTS_DIR / "features"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
ITEM_FEATURES_PATH = PROCESSED_DIR / "item_features.parquet"
RANKING_TRAIN_PATH = FEATURES_DIR / "ranking_train.parquet"
RANKING_TEST_PATH = FEATURES_DIR / "ranking_test.parquet"
SUMMARY_TRAIN_PATH = REPORTS_DIR / "ranking_train_summary.json"
SUMMARY_TEST_PATH = REPORTS_DIR / "ranking_test_summary.json"
DEFAULT_TOP_N = 100


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with a clear error if it is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found at {path}. Run `python -m scripts.prepare_data` first."
        )
    return pd.read_parquet(path)


def build_ranking_dataset(
    mode: str = "train",
    top_n: int = DEFAULT_TOP_N,
) -> None:
    """Build and persist ranking training or test candidate datasets.

    mode: "train" builds train->val dataset and saves ranking_train.parquet
    mode: "test" builds (train+val)->test candidate dataset and saves ranking_test.parquet
    """

    if mode not in {"train", "test"}:
        raise ValueError("mode must be one of 'train' or 'test'")

    train_df = load_parquet(TRAIN_PATH)
    val_df = load_parquet(VAL_PATH)
    item_features_df = load_parquet(ITEM_FEATURES_PATH)

    if mode == "train":
        ranking_df, feature_cols, summary = build_ranking_dataset_from_splits(
            history_df=train_df,
            target_df=val_df,
            item_features_df=item_features_df,
            candidate_top_n=top_n,
        )
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        ranking_df.to_parquet(RANKING_TRAIN_PATH, index=False)
        save_json({**summary, "feature_columns": feature_cols, "dataset_path": str(RANKING_TRAIN_PATH)}, SUMMARY_TRAIN_PATH)
        print(f"[ranking-dataset] saved {RANKING_TRAIN_PATH}")
        return

    # mode == test
    full_histories = pd.concat([train_df, val_df], ignore_index=True)
    test_df = load_parquet(TEST_PATH)
    ranking_df, feature_cols, summary = build_ranking_dataset_from_splits(
        history_df=full_histories,
        target_df=test_df,
        item_features_df=item_features_df,
        candidate_top_n=top_n,
    )
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ranking_df.to_parquet(RANKING_TEST_PATH, index=False)
    save_json({**summary, "feature_columns": feature_cols, "dataset_path": str(RANKING_TEST_PATH)}, SUMMARY_TEST_PATH)
    print(f"[ranking-dataset] saved {RANKING_TEST_PATH}")


def main() -> None:
    """CLI entrypoint for building train or test ranking datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Build ranking datasets (train or test candidates)")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    build_ranking_dataset(mode=args.mode, top_n=args.candidate_k)


if __name__ == "__main__":
    main()

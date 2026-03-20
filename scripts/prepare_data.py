"""Prepare RetailRocket interactions and item features for modeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_features import build_item_features
from src.data.load_data import load_category_tree, load_events, load_item_properties
from src.data.preprocess import preprocess_retailrocket_events
from src.utils.split import time_based_split


RAW_DATA_DIR = Path("data/raw/retailrocket")
PROCESSED_DATA_DIR = Path("data/processed")


def print_interaction_summary(interactions: pd.DataFrame) -> None:
    """Print concise summary statistics for interactions."""

    print(
        "[prepare] interactions"
        f" rows={len(interactions)}"
        f" users={interactions['user_id'].nunique()}"
        f" items={interactions['item_id'].nunique()}"
        f" min_ts={interactions['timestamp'].min()}"
        f" max_ts={interactions['timestamp'].max()}"
    )


def main() -> None:
    """Run the end-to-end RetailRocket data preparation pipeline."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] loading raw RetailRocket files from {RAW_DATA_DIR}")
    events = load_events(RAW_DATA_DIR)
    item_properties = load_item_properties(RAW_DATA_DIR)
    category_tree = load_category_tree(RAW_DATA_DIR)

    print(
        "[prepare] raw files"
        f" events={len(events)}"
        f" item_properties={len(item_properties)}"
        f" categories={len(category_tree)}"
    )

    interactions = preprocess_retailrocket_events(events)
    item_features = build_item_features(item_properties)
    train_df, val_df, test_df = time_based_split(interactions)

    print_interaction_summary(interactions)
    print(
        "[prepare] splits"
        f" train={len(train_df)}"
        f" val={len(val_df)}"
        f" test={len(test_df)}"
    )
    print(
        "[prepare] item_features"
        f" rows={len(item_features)}"
        f" items={item_features['item_id'].nunique()}"
    )

    output_paths = {
        "interactions": PROCESSED_DATA_DIR / "interactions.parquet",
        "train": PROCESSED_DATA_DIR / "train.parquet",
        "val": PROCESSED_DATA_DIR / "val.parquet",
        "test": PROCESSED_DATA_DIR / "test.parquet",
        "item_features": PROCESSED_DATA_DIR / "item_features.parquet",
    }

    interactions.to_parquet(output_paths["interactions"], index=False)
    train_df.to_parquet(output_paths["train"], index=False)
    val_df.to_parquet(output_paths["val"], index=False)
    test_df.to_parquet(output_paths["test"], index=False)
    item_features.to_parquet(output_paths["item_features"], index=False)

    for name, path in output_paths.items():
        print(f"[prepare] wrote {name} -> {path}")


if __name__ == "__main__":
    main()

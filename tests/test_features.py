"""Tests for RetailRocket feature preparation and splitting."""

from __future__ import annotations

import pandas as pd

from src.data.build_features import build_item_features, latest_item_properties
from src.utils.split import time_based_split


def test_latest_property_selection_logic() -> None:
    item_properties = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "itemid": [10, 10, 10],
            "property": ["categoryid", "categoryid", "brand"],
            "value": ["1", "2", "nike"],
        }
    )

    latest = latest_item_properties(item_properties)
    category_row = latest[latest["property"] == "categoryid"].iloc[0]

    assert category_row["value"] == "2"


def test_build_item_features_uses_latest_values() -> None:
    item_properties = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "itemid": [10, 10, 11, 11],
            "property": ["categoryid", "categoryid", "brand", "brand"],
            "value": ["1", "2", "a", "b"],
        }
    )

    features = build_item_features(item_properties, top_n_properties=5)

    assert "categoryid" in features.columns or "brand" in features.columns
    assert features.loc[features["item_id"] == 10, "categoryid"].iloc[0] == "2"
    assert features.loc[features["item_id"] == 11, "brand"].iloc[0] == "b"


def test_time_based_split_chronological_ordering() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "item_id": [11, 12, 13, 14, 15],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-04",
                ],
                utc=True,
            ),
        }
    )

    train_df, val_df, test_df = time_based_split(interactions, train_frac=0.6, val_frac=0.2)

    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()
    assert len(train_df) == 3
    assert len(val_df) == 1
    assert len(test_df) == 1

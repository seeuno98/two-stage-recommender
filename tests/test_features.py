"""Tests for feature preparation and ranking feature engineering."""

from __future__ import annotations

import pandas as pd

from src.data.build_features import build_item_features, latest_item_properties
from src.ranking.dataset import build_labeled_ranking_dataframe
from src.ranking.features import (
    add_ranking_features,
    build_item_feature_table,
    build_user_feature_table,
    encode_item_metadata,
)
from src.utils.split import time_based_split


def make_train_df() -> pd.DataFrame:
    """Create a small training frame for ranking feature tests."""

    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "item_id": [101, 102, 101, 103, 103],
            "event_type": ["view", "transaction", "view", "addtocart", "transaction"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-04T00:00:00Z",
                    "2024-01-05T00:00:00Z",
                ],
                utc=True,
            ),
            "event_weight": [1.0, 5.0, 1.0, 3.0, 5.0],
        }
    )


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


def test_item_popularity_features_computed_correctly() -> None:
    item_features = build_item_feature_table(make_train_df())
    row_103 = item_features[item_features["item_id"] == 103].iloc[0]

    assert row_103["item_popularity_count"] == 2
    assert row_103["item_popularity_weighted"] == 8.0
    assert row_103["item_event_addtocart_count"] == 1
    assert row_103["item_event_transaction_count"] == 1


def test_user_aggregate_features_computed_correctly() -> None:
    user_features = build_user_feature_table(make_train_df())
    row_1 = user_features[user_features["user_id"] == 1].iloc[0]

    assert row_1["user_train_interaction_count"] == 2
    assert row_1["user_train_unique_item_count"] == 2
    assert row_1["user_train_avg_event_weight"] == 3.0
    assert row_1["user_train_days_span"] == 2.0


def test_candidate_rows_preserve_labels_correctly() -> None:
    candidate_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [101, 105, 103],
            "popularity_rank": [1, 2, 1],
            "popularity_score": [5.0, 4.0, 3.0],
        }
    )

    labeled = build_labeled_ranking_dataframe(
        candidate_df,
        ground_truth={1: {105}, 2: {999}},
    )

    assert labeled["label"].tolist() == [0, 1, 0]


def test_missing_metadata_handled_safely() -> None:
    train_df = make_train_df()
    candidate_df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [103, 104],
            "label": [1, 0],
            "popularity_rank": [1, 2],
            "popularity_score": [8.0, 2.0],
        }
    )
    item_metadata = pd.DataFrame(
        {
            "item_id": [103],
            "categoryid": ["12"],
            "brand": ["acme"],
        }
    )

    features_df, feature_cols = add_ranking_features(candidate_df, train_df, item_metadata)
    row_104 = features_df[features_df["item_id"] == 104].iloc[0]

    assert set(feature_cols).issubset(features_df.columns)
    assert row_104["item_popularity_count"] == 0.0
    assert row_104["meta_categoryid"] == 0.0
    assert row_104["meta_brand_encoded"] == 0.0


def test_encode_item_metadata_produces_numeric_features() -> None:
    encoded = encode_item_metadata(
        pd.DataFrame(
            {
                "item_id": [1, 2],
                "categoryid": ["10", "20"],
                "brand": ["nike", None],
            }
        )
    )

    assert "meta_categoryid" in encoded.columns
    assert "meta_brand_encoded" in encoded.columns
    assert pd.api.types.is_numeric_dtype(encoded["meta_brand_encoded"])

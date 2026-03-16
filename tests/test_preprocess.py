"""Tests for data preprocessing."""

import pandas as pd

from src.data.preprocess import preprocess_interactions


def test_preprocess_interactions_sorts_and_parses_timestamps() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [100, 200],
            "timestamp": ["2024-01-02", "2024-01-01"],
        }
    )

    result = preprocess_interactions(interactions)

    assert list(result["user_id"]) == [2, 1]
    assert str(result["timestamp"].dtype).startswith("datetime64")

"""Tests for feature generation."""

import pandas as pd

from src.data.build_features import build_basic_interaction_features


def test_build_basic_interaction_features_adds_frequency_columns() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 11, 10],
        }
    )

    result = build_basic_interaction_features(interactions)

    assert "user_interaction_count" in result.columns
    assert "item_interaction_count" in result.columns
    assert result.loc[0, "user_interaction_count"] == 2
    assert result.loc[0, "item_interaction_count"] == 2

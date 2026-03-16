"""Feature building utilities for recommendation datasets."""

import pandas as pd


def build_basic_interaction_features(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Create simple frequency-based user and item interaction features."""

    features = interactions.copy()
    user_counts = features.groupby(user_col).size().rename("user_interaction_count")
    item_counts = features.groupby(item_col).size().rename("item_interaction_count")

    features = features.join(user_counts, on=user_col)
    features = features.join(item_counts, on=item_col)
    return features

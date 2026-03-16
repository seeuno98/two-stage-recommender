"""Preprocessing helpers for interaction data."""

import pandas as pd


def preprocess_interactions(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Apply a minimal cleaning pipeline to interaction data."""

    required_columns = {user_col, item_col, timestamp_col}
    missing_columns = required_columns.difference(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    processed = interactions.copy()
    processed = processed.dropna(subset=[user_col, item_col, timestamp_col])
    processed[timestamp_col] = pd.to_datetime(processed[timestamp_col], utc=True)
    processed = processed.sort_values(timestamp_col).reset_index(drop=True)
    return processed

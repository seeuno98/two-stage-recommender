"""Dataset splitting utilities."""

import pandas as pd


def time_based_train_test_split(
    interactions: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test partitions by timestamp order."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if timestamp_col not in interactions.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    ordered = interactions.sort_values(timestamp_col).reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_size))
    train = ordered.iloc[:split_index].reset_index(drop=True)
    test = ordered.iloc[split_index:].reset_index(drop=True)
    return train, test

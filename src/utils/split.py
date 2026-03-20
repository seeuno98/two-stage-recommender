"""Dataset splitting utilities."""

from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train, validation, and test sets chronologically."""

    if train_frac <= 0 or val_frac < 0:
        raise ValueError("train_frac must be > 0 and val_frac must be >= 0.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1.")
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    val_df = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test_df = ordered.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def time_based_train_test_split(
    interactions: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test partitions by timestamp order."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    train_df, _, test_df = time_based_split(
        interactions,
        train_frac=1 - test_size,
        val_frac=0.0,
        time_col=timestamp_col,
    )
    return train_df, test_df

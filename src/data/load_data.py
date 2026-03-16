"""Utilities for loading interaction datasets."""

from pathlib import Path

import pandas as pd


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load an interaction dataset from CSV or Parquet.

    Parameters
    ----------
    path:
        Path to the interaction dataset.

    Returns
    -------
    pd.DataFrame
        Loaded interaction records.

    Raises
    ------
    FileNotFoundError
        If the provided dataset path does not exist.
    ValueError
        If the file extension is unsupported.
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Interaction dataset not found: {dataset_path}")

    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)

    raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

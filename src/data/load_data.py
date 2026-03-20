"""Utilities for loading RetailRocket and generic interaction datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RETAILROCKET_FILES = {
    "events": "events.csv",
    "item_properties_part1": "item_properties_part1.csv",
    "item_properties_part2": "item_properties_part2.csv",
    "category_tree": "category_tree.csv",
}


def _resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve a data directory into a concrete path."""

    return Path(data_dir)


def _require_file(path: Path, description: str) -> Path:
    """Validate that an expected file exists."""

    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}, but it was not found.")
    return path


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load an interaction dataset from CSV or Parquet."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Interaction dataset not found: {dataset_path}")

    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)

    raise ValueError(f"Unsupported file format: {dataset_path.suffix}")


def load_events(data_dir: str | Path) -> pd.DataFrame:
    """Load RetailRocket interaction events."""

    raw_dir = _resolve_data_dir(data_dir)
    events_path = _require_file(
        raw_dir / RETAILROCKET_FILES["events"],
        "RetailRocket events.csv",
    )
    return pd.read_csv(events_path)


def load_item_properties(data_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate RetailRocket item property files."""

    raw_dir = _resolve_data_dir(data_dir)
    part1_path = _require_file(
        raw_dir / RETAILROCKET_FILES["item_properties_part1"],
        "RetailRocket item_properties_part1.csv",
    )
    part2_path = _require_file(
        raw_dir / RETAILROCKET_FILES["item_properties_part2"],
        "RetailRocket item_properties_part2.csv",
    )
    part1 = pd.read_csv(part1_path)
    part2 = pd.read_csv(part2_path)
    return pd.concat([part1, part2], ignore_index=True)


def load_category_tree(data_dir: str | Path) -> pd.DataFrame:
    """Load the RetailRocket category tree."""

    raw_dir = _resolve_data_dir(data_dir)
    category_tree_path = _require_file(
        raw_dir / RETAILROCKET_FILES["category_tree"],
        "RetailRocket category_tree.csv",
    )
    return pd.read_csv(category_tree_path)

"""Feature building utilities for recommendation datasets."""

from __future__ import annotations

import re

import pandas as pd


PROPERTY_NAME_PATTERN = re.compile(r"[^0-9a-zA-Z_]+")


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


def parse_item_property_timestamps(item_properties: pd.DataFrame) -> pd.DataFrame:
    """Parse RetailRocket item property timestamps from milliseconds to UTC datetimes."""

    parsed = item_properties.copy()
    parsed["timestamp"] = pd.to_datetime(parsed["timestamp"], unit="ms", utc=True)
    return parsed


def latest_item_properties(item_properties: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest property value per item and property."""

    required_columns = {"timestamp", "itemid", "property", "value"}
    missing_columns = required_columns.difference(item_properties.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Item properties are missing required columns: {missing}")

    parsed = parse_item_property_timestamps(item_properties)
    latest = (
        parsed.sort_values(["itemid", "property", "timestamp"])
        .drop_duplicates(subset=["itemid", "property"], keep="last")
        .rename(columns={"itemid": "item_id"})
        .reset_index(drop=True)
    )
    return latest


def _sanitize_property_name(property_name: str) -> str:
    """Sanitize a property name for use as a dataframe column."""

    sanitized = PROPERTY_NAME_PATTERN.sub("_", property_name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_").lower()
    return sanitized or "unknown_property"


def build_item_features(
    item_properties: pd.DataFrame,
    top_n_properties: int = 20,
) -> pd.DataFrame:
    """Build a lightweight item feature table from latest item properties."""

    latest = latest_item_properties(item_properties)

    property_counts = latest["property"].value_counts()
    selected_properties = property_counts.head(top_n_properties).index.tolist()
    selected = latest[latest["property"].isin(selected_properties)].copy()
    selected["feature_name"] = selected["property"].map(_sanitize_property_name)

    feature_table = (
        selected.pivot(index="item_id", columns="feature_name", values="value")
        .reset_index()
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    return feature_table

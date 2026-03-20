"""Preprocessing helpers for recommendation datasets."""

from __future__ import annotations

import pandas as pd


EVENT_WEIGHT_MAP = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}
STANDARD_EVENT_COLUMNS = {
    "visitorid": "user_id",
    "itemid": "item_id",
    "event": "event_type",
    "timestamp": "timestamp",
}
REQUIRED_STANDARD_COLUMNS = ["user_id", "item_id", "event_type", "timestamp"]


def rename_retailrocket_event_columns(events: pd.DataFrame) -> pd.DataFrame:
    """Rename RetailRocket event columns into the project standard schema."""

    required_columns = set(STANDARD_EVENT_COLUMNS)
    missing = required_columns.difference(events.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"RetailRocket events are missing columns: {missing_columns}")
    return events.rename(columns=STANDARD_EVENT_COLUMNS)


def parse_event_timestamps(events: pd.DataFrame) -> pd.DataFrame:
    """Parse RetailRocket timestamps from milliseconds to UTC datetimes."""

    parsed = events.copy()
    parsed["timestamp"] = pd.to_datetime(parsed["timestamp"], unit="ms", utc=True)
    return parsed


def add_event_weights(events: pd.DataFrame) -> pd.DataFrame:
    """Map event types to numeric implicit-feedback weights."""

    weighted = events.copy()
    weighted["event_weight"] = weighted["event_type"].map(EVENT_WEIGHT_MAP)
    return weighted


def drop_invalid_rows(events: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing required fields or event weights."""

    cleaned = events.dropna(subset=REQUIRED_STANDARD_COLUMNS + ["event_weight"])
    return cleaned


def deduplicate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate interaction rows."""

    return events.drop_duplicates()


def preprocess_retailrocket_events(events: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw RetailRocket interactions into a standardized schema."""

    processed = rename_retailrocket_event_columns(events)
    processed = parse_event_timestamps(processed)
    processed = add_event_weights(processed)
    processed = drop_invalid_rows(processed)
    processed = deduplicate_events(processed)
    processed = processed[REQUIRED_STANDARD_COLUMNS + ["event_weight"]]
    processed["user_id"] = processed["user_id"].astype("int64")
    processed["item_id"] = processed["item_id"].astype("int64")
    processed["event_type"] = processed["event_type"].astype(str)
    processed["event_weight"] = processed["event_weight"].astype(float)
    processed = processed.sort_values("timestamp").reset_index(drop=True)
    return processed


def preprocess_interactions(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Apply minimal generic interaction cleaning."""

    required_columns = {user_col, item_col, timestamp_col}
    missing_columns = required_columns.difference(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    processed = interactions.copy()
    processed = processed.dropna(subset=[user_col, item_col, timestamp_col])
    processed[timestamp_col] = pd.to_datetime(processed[timestamp_col], utc=True)
    processed = processed.drop_duplicates()
    processed = processed.sort_values(timestamp_col).reset_index(drop=True)
    return processed

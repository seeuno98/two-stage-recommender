"""Tests for RetailRocket preprocessing."""

from __future__ import annotations

import pandas as pd

from src.data.preprocess import preprocess_retailrocket_events


def test_event_weight_mapping() -> None:
    events = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "visitorid": [1, 1, 1],
            "event": ["view", "addtocart", "transaction"],
            "itemid": [10, 10, 10],
        }
    )

    result = preprocess_retailrocket_events(events)

    assert result["event_weight"].tolist() == [1.0, 3.0, 5.0]


def test_null_row_dropping() -> None:
    events = pd.DataFrame(
        {
            "timestamp": [1, 2],
            "visitorid": [1, None],
            "event": ["view", "view"],
            "itemid": [10, 11],
        }
    )

    result = preprocess_retailrocket_events(events)

    assert len(result) == 1
    assert result.iloc[0]["user_id"] == 1


def test_duplicate_removal() -> None:
    events = pd.DataFrame(
        {
            "timestamp": [1, 1],
            "visitorid": [1, 1],
            "event": ["view", "view"],
            "itemid": [10, 10],
        }
    )

    result = preprocess_retailrocket_events(events)

    assert len(result) == 1
